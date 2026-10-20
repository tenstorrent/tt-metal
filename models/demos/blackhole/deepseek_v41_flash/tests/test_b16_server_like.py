# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Server-like slot isolation probe: the vLLM adapter (``DeepseekV41ForCausalLM.initialize_vllm_model``: the production B=16 / 262144 / spec k=5 build with warm-up traces) driven
directly with the plugin's call protocol (prefill_forward with empty_slots, [W,1] decode, propose_draft_tokens, wide verify with spec_mode / num_valid_drafts / accepted_counts),
without vLLM or its scheduler. A single-request or multi-request simulation (``Sim``) replays the 4,934-token tool-calling prompt (probe_t22_k3_ids.json) in chosen slots.

Interactive mode (always): after the build the process executes the python files that appear in ISO_CMD_DIR (sorted by name, each once) in a namespace holding ``Sim``,
``model`` (the adapter), ``ids`` (the prompt), ``md``, ``log``; a file called ``stop`` ends it. Env: ISO_B (16), ISO_CTX (262144), ISO_IDS.
"""

import json
import os
import sys
import time
import traceback
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.model_args import CKPT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = os.environ.get("ISO_IDS", "/mnt/tt-data/ssinghal/dsv4-logs/evals_gpqa_tau3/probe_t22_k3_ids.json")
GOOD = "omar_rossi_1241"


def _stub_plugin():
    """tt-vllm-plugin is not installed in the tt-metal venv: the two result classes the adapter returns."""
    if "vllm_tt_plugin.spec_decode" in sys.modules:
        return

    @dataclass
    class VerifyOutput:
        spec_mode: str
        argmax_ids: object = None

    @dataclass
    class DraftOutput:
        draft_token_ids: object
        num_valid: object = None

    pkg = types.ModuleType("vllm_tt_plugin")
    sub = types.ModuleType("vllm_tt_plugin.spec_decode")
    sub.VerifyOutput, sub.DraftOutput = VerifyOutput, DraftOutput
    pkg.spec_decode = sub
    sys.modules["vllm_tt_plugin"], sys.modules["vllm_tt_plugin.spec_decode"] = pkg, sub


def _hf_config():
    cfg = json.load(open(os.path.join(HERE, "vllm_config", "config.json")))
    ns = SimpleNamespace(**cfg)
    ns._name_or_path = CKPT
    ns.to_dict = lambda: cfg
    return ns


class Sim:
    """Plugin-protocol driver. ``reqs``: {logical slot: token ids}. ``mode``: 'spec' (propose + wide verify, the production flow) or 'plain' (ordinary [W,1] decode steps, no drafts)."""

    def __init__(self, model, W=16):
        self.m, self.W, self.K = model, W, model.spec_k

    @staticmethod
    def sp(n):
        return SimpleNamespace(temperature=[0.0] * n, top_k=[1] * n, top_p=[1.0] * n, seed=[None] * n)

    def prefill(self, reqs, together=True):
        out = {}
        groups = [list(reqs)] if together else [[s] for s in reqs]
        for g in groups:
            lens = [len(reqs[s]) for s in g]
            toks = torch.zeros(len(g), max(lens), dtype=torch.int32)
            for i, s in enumerate(g):
                toks[i, : lens[i]] = torch.tensor(reqs[s], dtype=torch.int32)
            f = self.m.prefill_forward(
                toks,
                prompt_lens=lens,
                empty_slots=list(g),
                enable_trace=True,
                sampling_params=self.sp(len(g)),
                start_pos=torch.zeros(len(g), dtype=torch.int64),
            )
            for i, s in enumerate(g):
                out[s] = int(f[i])
        return out

    def run(self, reqs, ntok=80, mode="spec", together=True, first_via_propose=True):
        """-> {slot: committed token list (first token included)}, {slot: phys user}"""
        m, W, K = self.m, self.W, self.K
        for s in reqs:
            m.release_request(s)
        first = self.prefill(reqs, together)
        com = {s: [first[s]] for s in reqs}
        pos = {s: len(reqs[s]) for s in reqs}  # position of the last committed token (the one fed next)
        phys = {s: m.slots.phys[s] for s in reqs}
        has_drafts = False
        stats = {"rounds": 0, "acc": 0}
        t0 = time.time()
        while min(len(c) for c in com.values()) < ntok:
            tokens = torch.zeros(W, 1 + (K if has_drafts and mode == "spec" else 0), dtype=torch.int32)
            start = torch.full((W, tokens.shape[1]), -1, dtype=torch.int64)
            for s in reqs:
                tokens[s, 0] = com[s][-1]
                start[s] = torch.arange(pos[s], pos[s] + tokens.shape[1])
            if mode == "plain" or not has_drafts and not first_via_propose:
                out = m.decode_forward(tokens[:, :1], start[:, 0], sampling_params=self.sp(W))
                for s in reqs:
                    com[s].append(int(out[s, 0]))
                    pos[s] += 1
                has_drafts = False
                if mode == "spec":
                    pass
                continue
            if not has_drafts:  # first verify: drafts proposed right after the prefill step (production flow)
                committed = torch.full((W, 1 + K), -1, dtype=torch.int32)
                cp = torch.full((W, 1 + K), -1, dtype=torch.int64)
                cnt = torch.ones(W, dtype=torch.int32)
                for s in reqs:
                    committed[s, 0], cp[s, 0] = com[s][-1], pos[s]
                d = m.propose_draft_tokens(K, committed, cp, cnt)
                drafts, nv = d.draft_token_ids, d.num_valid
                has_drafts = True
                tokens = torch.zeros(W, 1 + K, dtype=torch.int32)
                start = torch.full((W, 1 + K), -1, dtype=torch.int64)
                for s in reqs:
                    tokens[s, 0] = com[s][-1]
                    tokens[s, 1:] = drafts[s]
                    start[s] = torch.arange(pos[s], pos[s] + 1 + K)
            v = m.decode_forward(
                tokens,
                start,
                spec_mode="argmax_ids",
                num_valid_drafts=nv.clone(),
                accepted_counts=torch.ones(W, dtype=torch.int32),
                sampling_params=self.sp(W),
            )
            a = v.argmax_ids
            committed = torch.full((W, 1 + K), -1, dtype=torch.int32)
            cp = torch.full((W, 1 + K), -1, dtype=torch.int64)
            cnt = torch.ones(W, dtype=torch.int32)
            for s in reqs:
                mm = 0
                if int(nv[s]) > 0:
                    while mm < K and int(tokens[s, 1 + mm]) == int(a[s, mm]):
                        mm += 1
                new = [int(a[s, j]) for j in range(mm + 1)]
                com[s].extend(new)
                stats["acc"] += mm
                committed[s, : mm + 1] = torch.tensor(new, dtype=torch.int32)
                pos[s] += mm + 1
                cp[s, : mm + 1] = torch.arange(pos[s] - mm - 1, pos[s])
                cnt[s] = mm + 1
            stats["rounds"] += 1
            d = m.propose_draft_tokens(K, committed, cp, cnt)
            drafts, nv = d.draft_token_ids, d.num_valid
        stats["secs"] = round(time.time() - t0, 1)
        self.stats = stats
        return com, phys

    def report(self, name, com, phys, tok, U=4):
        res = {}
        for s, c in com.items():
            text = tok.decode(c)
            ok = GOOD in text
            res[s] = ok
            print(
                f"SIM[{name}] slot {s} (phys {phys[s]}, row {phys[s] // U}, in-row {phys[s] % U}) ok={ok} first={c[:12]} text={text[:150]!r}",
                flush=True,
            )
        print(
            f"SIM[{name}] stats {getattr(self, 'stats', None)} summary {json.dumps({str(k): v for k, v in res.items()})}",
            flush=True,
        )
        return res


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "800000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
def test_server_like(mesh_device, device_params):
    from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM

    _stub_plugin()
    md = mesh_device
    log = lambda s: print(s, flush=True)
    B = int(os.environ.get("ISO_B", "16"))
    ctx = int(os.environ.get("ISO_CTX", "262144"))
    model = DeepseekV41ForCausalLM.initialize_vllm_model(_hf_config(), md, B, ctx)
    log("MODEL READY")
    ids = json.load(open(IDS))["ids"]
    tok = model.generator.tokenizer
    cmd_dir = os.environ["ISO_CMD_DIR"]
    os.makedirs(cmd_dir, exist_ok=True)
    ns = dict(globals())
    ns.update(md=md, model=model, ids=ids, tok=tok, log=log, Sim=Sim, torch=torch, ttnn=ttnn)
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
