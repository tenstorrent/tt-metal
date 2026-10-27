# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The DSpark drafter on the mesh (``tt/v41/dspark_drafter.py``) against the vendored reference (speculative
decode device gate).

The reference is ``real_spec`` with the backbone of layers 20, 36-39 synthetic and the three ``mtp.*`` units
from the
checkpoint (as ``test_dspark.py``'s ``real_ckpt``), greedy. It prefills ``V41_DSD_PROMPT`` tokens of real text
and seeds
its rings (``forward_spec`` at start_pos 0); the device drafter is built from the checkpoint through the
loader
(the routed experts converted into the drafter's own cache directory on the first run) with the reference's
tied
embedding / head, and takes the reference's rings. Then, for ``V41_DSD_STEPS`` positions, both sides run one
draft
call on the reference's tap of the position, teacher-forced on the text's next token, and the test compares:
stage
0's ``main_x`` and every stage's ring write (bit-exact rate and PCC), every stage's output streams on the
block's
rows (PCC), the five rows' logits after the Markov loop (PCC) and the drafted ids (exact matches), plus the
determinism of a repeated call. It reports the draft call's wall time (eager; the Markov loop on the host and
the
logits readback apart) and the DRAM a chip before / after the drafter: ``V41_DSD_OUT`` takes the JSON record.
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import WEIGHT_CACHE, routed_expert_dtype
from models.demos.deepseek_v3_d_p.tt.v41.dspark_drafter import DSPARK_MARKOV, RING, ROWS, TtDSparkDrafter
from models.demos.deepseek_v3_d_p.tt.v41.head import TtV41Embedding
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint
from tests.ttnn.utils_for_testing import comp_pcc

REAL_LAYERS = (20, 36, 37, 38, 39)
DSPARK_UNITS = tuple(f"mtp.{k}" for k in range(C.NUM_DSPARK_LAYERS))
TRACE_REGION = 256 * 1024**2

MAIN_X_PCC = 0.999
RING_PCC = 0.998
# the stages' streams and the logits against the bf16 reference with bfp8 / bfp4 routed experts (measured:
# stage 2 at 0.95 to 0.97 with bfp4, the logits 0.98 to 0.9999; the drafted ids agree)
STREAM_PCC = {"BFLOAT8_B": 0.97, "BFLOAT4_B": 0.94}
LOGITS_PCC = 0.97


def _pcc(a, b) -> float:
    return float(comp_pcc(a.float(), b.float(), 0.0)[1])


def _dram(mesh_device) -> dict:
    """A chip's DRAM: bytes allocated, bytes free (the live margin) and the largest free block, from the
    allocator."""
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        banks = int(view.num_banks)
        return {
            "allocated": int(view.total_bytes_allocated_per_bank) * banks,
            "free": int(view.total_bytes_free_per_bank) * banks,
            "largest_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
            "banks": banks,
        }
    except Exception as e:
        logger.warning(f"DRAM statistics unavailable: {e}")
        return {}


def _first(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])[0, 0]


def _streams_to_host(t, mesh_device, hc: int, hidden: int) -> torch.Tensor:
    """Device streams [1, 1, ROWS, hc * hidden/tp] fp32 per chip (chip t: its hidden slice of every copy) ->
    [ROWS,
    hc, hidden] from mesh row 0."""
    sp, tp = tuple(mesh_device.shape)
    shards = ttnn.get_device_tensors(t)
    pieces = [ttnn.to_torch(shards[j])[0, 0] for j in range(tp)]  # mesh row 0's chips
    x = torch.cat(pieces, dim=-1)  # [ROWS, tp * hc * hidden/tp] in (chip, copy, j) order
    return x.reshape(ROWS, tp, hc, hidden // tp).permute(0, 2, 1, 3).reshape(ROWS, hc, hidden)


@pytest.mark.timeout(int(os.environ.get("V41_DSD_TIMEOUT", 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_dspark_drafter(mesh_device, device_params):
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    prompt = int(os.environ.get("V41_DSD_PROMPT", 160))
    steps = int(os.environ.get("V41_DSD_STEPS", 3))
    timed = int(os.environ.get("V41_DSD_TIMED", 5))
    stages = int(os.environ.get("V41_DSD_STAGES", C.NUM_DSPARK_LAYERS))
    assert prompt >= RING, "the reference attends min(128, p + 1) ring slots; keep the ring full"
    sp, tp = tuple(mesh_device.shape)
    dtype = routed_expert_dtype()
    record = {
        "prompt": prompt,
        "steps": steps,
        "stages": stages,
        "expert_dtype": dtype.name,
        "mesh": [sp, tp],
        "flags": {k: v for k, v in os.environ.items() if k.startswith(("V41_", "TT_V41_"))},
    }

    # --- the reference, prefilled and seeded ---
    t0 = time.time()
    spec = orc.real_spec(REAL_LAYERS, 512, dspark=True, checkpoint=ckpt.root, checkpoint_units=DSPARK_UNITS)
    model = orc.build_reference(spec)
    model.temperature = 0.0
    for block in model.mtp:
        block.temperature = 0.0
    # the tied tables go to the device in bf16: hold the reference to the same values (a synthetic unit's are
    # fp32)
    model.head.weight.data = model.head.weight.data.to(torch.bfloat16).float()
    model.embed.weight.data = model.embed.weight.data.to(torch.bfloat16).to(model.embed.weight.dtype)
    record["reference_build_s"] = round(time.time() - t0, 1)
    tokens = orc.text_tokens(prompt + steps + 2)
    t0 = time.time()
    with torch.inference_mode(), v41.set_dtype(torch.bfloat16):
        out_ids, _, main_hidden = model(tokens[:, :prompt], 0)
        model.forward_spec(out_ids, main_hidden, 0)
    record["reference_prefill_s"] = round(time.time() - t0, 1)
    rings = [block.attn.window_kv_cache[0].clone() for block in model.mtp]

    captured = {}

    def stage_hook(k):
        def hook(module, args, output):
            captured[f"stage{k}"] = output[0].detach().clone()  # [1, block, hc, dim] bf16

        return hook

    def main_x_hook(module, args, output):
        captured["main_x"] = output.detach().clone()  # [1, 1, dim] bf16

    for k, block in enumerate(model.mtp):
        block.register_forward_hook(stage_hook(k))
    model.mtp[0].main_norm.register_forward_hook(main_x_hook)

    # --- the device drafter ---
    record["dram_before"] = _dram(mesh_device)
    t0 = time.time()
    embedding = TtV41Embedding(mesh_device, C, model.embed.weight.detach().to(torch.bfloat16))
    record["dram_after_embedding"] = _dram(mesh_device)
    drafter = TtDSparkDrafter(
        mesh_device,
        C,
        ckpt,
        embedding,
        model.head.weight.detach().to(torch.bfloat16),
        WEIGHT_CACHE,
        dtype,
        stages=stages,
    )
    record["drafter_build_s"] = round(time.time() - t0, 1)
    record["drafter_build_detail_s"] = drafter.build_s
    record["cache_dir"] = str(drafter.cache_dir)
    record["dram_after_drafter"] = _dram(mesh_device)
    if record["dram_after_drafter"] and record["dram_after_embedding"]:
        # the stages, the seeder and the head (the tied embedding is the backbone's in serving, the head too)
        record["drafter_dram_bytes"] = (
            record["dram_after_drafter"]["allocated"] - record["dram_after_embedding"]["allocated"]
        )
    drafter.load_rings(rings[:stages])
    hc, hidden = C.HC_MULT, C.EMB_SIZE

    results, exact, per_step = {}, {}, []
    walls = []
    for i in range(steps):
        p = prompt + i
        tok = int(tokens[0, p + 1])
        t0 = time.time()
        with torch.inference_mode(), v41.set_dtype(torch.bfloat16):
            _, _, main_hidden = model(tokens[:, p : p + 1], p)
            ref_out, ref_logits, _ = model.forward_spec(tokens[0, p + 1 : p + 2], main_hidden, p)
        ref_s = time.time() - t0
        ref_rings = [block.attn.window_kv_cache[0, p % RING].clone() for block in model.mtp]

        inputs = drafter.to_device(drafter.host_inputs(tok, main_hidden[0, 0], p))
        stage_out = {}

        def on_stage(k, x):
            stage_out[k] = _streams_to_host(x, mesh_device, hc, hidden)

        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        if DSPARK_MARKOV == "device":
            ids_dev, logits_dev, main_x = drafter.draft_ids(inputs, on_stage=on_stage)
            device_ids = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(ids_dev)[0]).flatten()[:5]]
        else:
            logits_dev, main_x = drafter.draft(inputs, on_stage=on_stage)
            device_ids = None
        rows = drafter.logits_rows(logits_dev)
        wall = time.time() - t0
        t0 = time.time()
        out, rows = drafter.markov_loop(tok, rows)
        markov_s = time.time() - t0

        tag = f"p{p}"
        dev_main_x = _first(main_x)[0]
        results[f"{tag}_main_x"] = _pcc(captured["main_x"][0, 0], dev_main_x)
        exact[f"{tag}_main_x_bits"] = float((captured["main_x"][0, 0].float() == dev_main_x.float()).float().mean())
        for k in range(stages):
            dev_ring = drafter.ring_rows(k)[p % RING]
            results[f"{tag}_ring{k}"] = _pcc(ref_rings[k], dev_ring)
            exact[f"{tag}_ring{k}_bits"] = float((ref_rings[k].float() == dev_ring.float()).float().mean())
            ref_x = captured[f"stage{k}"][0]  # [block, hc, dim]
            results[f"{tag}_stage{k}"] = _pcc(ref_x, stage_out[k][: ref_x.shape[0]])
        if stages == C.NUM_DSPARK_LAYERS:
            results[f"{tag}_logits"] = _pcc(ref_logits[0], rows)
            ref_ids = ref_out[0, 1:].tolist()
            dev_ids = out[1:].tolist()
            exact[f"{tag}_ids"] = sum(a == b for a, b in zip(ref_ids, dev_ids)) / len(ref_ids)
            exact[f"{tag}_first_id"] = float(ref_ids[0] == dev_ids[0])
            if device_ids is not None:  # the in-graph Markov loop against the host one (bf16 candidates: ties)
                exact[f"{tag}_device_loop_ids"] = sum(a == b for a, b in zip(device_ids, dev_ids)) / len(dev_ids)
        else:
            ref_ids, dev_ids = [], []
        entry = {
            "p": p,
            "token": tok,
            "ref_s": round(ref_s, 2),
            "draft_wall_s": round(wall, 4),
            "markov_host_s": round(markov_s, 4),
            "ref_ids": ref_ids,
            "dev_ids": dev_ids,
            "device_loop_ids": device_ids,
        }
        per_step.append(entry)
        logger.info(f"drafter step {json.dumps(entry)}")
        for t in inputs.values():
            for u in t if isinstance(t, tuple) else (t,):
                ttnn.deallocate(u)

    # --- determinism and the timed calls (the last position again) ---
    p = prompt + steps - 1
    tok = int(tokens[0, p + 1])
    inputs = drafter.to_device(drafter.host_inputs(tok, main_hidden[0, 0], p))
    ids = []
    for _ in range(2):
        logits_dev, _ = drafter.draft(inputs)
        out, _ = drafter.markov_loop(tok, drafter.logits_rows(logits_dev))
        ids.append(out.tolist())
    exact["deterministic"] = float(ids[0] == ids[1])
    for _ in range(timed):
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        logits_dev, _ = drafter.draft(inputs)
        ttnn.synchronize_device(mesh_device)
        t1 = time.time()
        rows = drafter.logits_rows(logits_dev)
        t2 = time.time()
        drafter.markov_loop(tok, rows)
        t3 = time.time()
        entry = {"device_wall_ms": (t1 - t0) * 1e3, "readback_ms": (t2 - t1) * 1e3, "markov_ms": (t3 - t2) * 1e3}
        if DSPARK_MARKOV == "device":
            ttnn.synchronize_device(mesh_device)
            t0 = time.time()
            ids_dev, logits_dev, _ = drafter.draft_ids(inputs)
            ttnn.synchronize_device(mesh_device)
            t1 = time.time()
            ttnn.to_torch(ttnn.get_device_tensors(ids_dev)[0])
            t2 = time.time()
            ttnn.deallocate(logits_dev)
            entry["device_loop_wall_ms"], entry["device_loop_readback_ms"] = (t1 - t0) * 1e3, (t2 - t1) * 1e3
        walls.append(entry)
    record["timed"] = [{k: round(v, 3) for k, v in w.items()} for w in walls]
    record["draft_device_wall_ms_min"] = round(min(w["device_wall_ms"] for w in walls), 3)
    record["draft_device_wall_ms_median"] = round(sorted(w["device_wall_ms"] for w in walls)[len(walls) // 2], 3)
    record["readback_ms_min"] = round(min(w["readback_ms"] for w in walls), 3)
    record["markov_ms_min"] = round(min(w["markov_ms"] for w in walls), 3)
    record["dram_after_steps"] = _dram(mesh_device)
    record["steps_detail"] = per_step
    record["pcc"] = {k: round(v, 6) for k, v in results.items()}
    record["exact"] = exact
    logger.info("V41_DSD_RESULT " + json.dumps(record))
    out_path = os.environ.get("V41_DSD_OUT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(record, f, indent=1)
    print(f"dspark drafter pcc: {record['pcc']}")
    print(f"dspark drafter exact: {exact}")
    print(f"dspark drafter draft call: {record['timed']}")
    for key, value in results.items():
        bar = (
            MAIN_X_PCC
            if key.endswith("main_x")
            else RING_PCC
            if "_ring" in key
            else LOGITS_PCC
            if key.endswith("logits")
            else STREAM_PCC[dtype.name]
        )
        assert value >= bar, (key, value, bar)
    assert exact["deterministic"], "two draft calls on the same inputs differ"
