# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Server-like harness of image input: the vLLM adapter (``DeepseekV41ForCausalLM.initialize_vllm_model``: 40 layers, B=16, spec k=5, vision tower, warm-up traces) driven directly
(no vLLM), to bisect run-to-run differences that only show in the full server.

Experiments (VIS_EXP, comma list): text (repeat a text prefill: first-token logits + prefill taps checksum must be identical every time), tower (the tower alone: replicas of every
mesh row must agree, the rows identical run to run), image (tower + image prefill repeated), churn (text prefill after allocation churn).
Env: DSV41_VLLM_SPEC_K=5 DSV41_VISION=1 DSV41_ENGRAM_RAM=1 DSV41_POOL_DTYPE=fp8 DSV41_TRACE_REGION=1600000000 (set by the launcher).
"""

import json
import os
import traceback
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.model_args import CKPT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_RUNS = int(os.environ.get("VIS_RUNS", "6"))


def _hf_config():
    cfg = json.load(open(os.path.join(HERE, "vllm_config", "config.json")))
    ns = SimpleNamespace(**cfg)
    ns._name_or_path = CKPT
    ns.to_dict = lambda: cfg
    return ns


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 1600000000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
def test_server_like(mesh_device):
    import traceback

    try:
        _body(mesh_device)
    except BaseException:
        traceback.print_exc()
        raise


@torch.no_grad()
def _body(md):
    from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP
    from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM

    log = lambda s: print(s, flush=True)
    model = DeepseekV41ForCausalLM.initialize_vllm_model(_hf_config(), md, 16, int(os.environ.get("VIS_CTX", "32768")))
    log("MODEL READY")
    exps = os.environ.get("VIS_EXP", "text,tower,image").split(",")
    g = torch.Generator().manual_seed(5)
    text = torch.randint(1000, 100000, (1, 200), generator=g).to(torch.int32)

    def taps_sum():
        t = model.m.sink.taps
        h = t.chunk_hidden(0)
        s = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float()
        ttnn.deallocate(h)
        return round(s.sum().item(), 4), round(s.abs().sum().item(), 1)

    def prefill(tokens, **kw):
        model.release_request(0)
        out = model.prefill_forward(
            tokens, prompt_lens=[tokens.shape[1]], empty_slots=[0], enable_trace=True, sampling_params=None, **kw
        )
        return out.float().reshape(-1), taps_sum()

    def report(name, runs):
        lg = [r[0] for r in runs]
        same = [bool(torch.equal(lg[0], x)) for x in lg]
        log(f"{name}: logits equal to run 0: {same}; taps {[r[1] for r in runs]}")

    if os.environ.get("VIS_LOOP") == "1":
        # interactive mode: the process keeps the built model and executes the python files that appear in VIS_CMD_DIR (sorted by name, each once) in this namespace; a file called "stop" ends it.
        import time

        cmd_dir = os.environ["VIS_CMD_DIR"]
        os.makedirs(cmd_dir, exist_ok=True)
        ns = dict(globals())
        ns.update(
            md=md,
            model=model,
            text=text,
            prefill=prefill,
            report=report,
            taps_sum=taps_sum,
            log=log,
            VP=VP,
            N_RUNS=N_RUNS,
        )
        done = set()
        log("LOOP READY")
        while "stop" not in os.listdir(cmd_dir):
            for f in sorted(x for x in os.listdir(cmd_dir) if x.endswith(".py") and x not in done):
                done.add(f)
                log(f"CMD_START {f}")
                try:
                    exec(compile(open(os.path.join(cmd_dir, f)).read(), f, "exec"), ns)
                except BaseException:
                    traceback.print_exc()
                log(f"CMD_DONE {f}")
            time.sleep(2)
        log("DONE")
        return
    if "text" in exps:
        report("TEXT prefill x%d" % N_RUNS, [prefill(text) for _ in range(N_RUNS)])
    if "tower" in exps:
        va = model.vision.a
        pil = __import__("PIL.Image", fromlist=["Image"]).Image
        import numpy as np

        rng = np.random.RandomState(0)
        im = pil.fromarray(rng.randint(0, 255, (13, 17, 3)).astype(np.uint8)).resize((560, 420), pil.BICUBIC)
        patches, n_h, n_w, h, w = VP.patchify_pil(im, va)
        outs = [model.vision.encode(patches, n_h, n_w) for _ in range(N_RUNS)]
        log("TOWER rows equal to run 0: %s" % [bool(torch.equal(outs[0], o)) for o in outs])
    if "image" in exps:
        va = model.vision.a
        pil = __import__("PIL.Image", fromlist=["Image"]).Image
        import numpy as np

        rng = np.random.RandomState(0)
        im = pil.fromarray(rng.randint(0, 255, (13, 17, 3)).astype(np.uint8)).resize((560, 420), pil.BICUBIC)
        patches, n_h, n_w, h, w = VP.patchify_pil(im, va)
        span = h * (w + 1) + 2
        toks = torch.cat(
            [text[:, :50], torch.full((1, span), va.image_token_id, dtype=torch.int32), text[:, 50:]], dim=1
        )
        grid = torch.tensor([[1, n_h, n_w]])
        pv = patches.reshape(n_h * n_w, -1)
        model.vbook.cache_max = 0
        report(
            "IMAGE prefill x%d" % N_RUNS,
            [prefill(toks, pixel_values=[[pv]], image_grid_thw=[[grid]]) for _ in range(N_RUNS)],
        )
    if "churn" in exps:
        junk = [
            ttnn.from_torch(
                torch.randn(1, 1, 8192, 8192).to(torch.bfloat16),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            for _ in range(8)
        ]
        ttnn.synchronize_device(md)
        for t in junk:
            ttnn.deallocate(t)
        report("TEXT prefill after DRAM churn", [prefill(text) for _ in range(3)])
    log("DONE")
