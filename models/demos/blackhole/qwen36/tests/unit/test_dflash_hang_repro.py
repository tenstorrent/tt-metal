# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal reproduction: WHICH generation after a capture hangs?

Four device resets have gone into a hang whose only characterisation is "a generation after
enable_traced_verify() in a process that keeps going". test_dflash_traced_throughput.py never hits
it because it enables the trace, runs exactly ONE generation, and ends.

Tiny prompts and 4-token generations, logged one at a time, so the transition that stops is visible
rather than inferred. If generation 2-after-capture hangs while 1 succeeds, the trigger is "more
than one traced generation per capture" -- which would make it a real serving blocker, not a test
artifact.

MEASURED 2026-09-15 under TT_METAL_WATCHER=10 (NOC_SANITIZE/RING_BUFFER/STACK_USAGE/ASSERT/PAUSE/
CB_SANITIZE/ETH_LINK_STATUS all disabled -- leaving them on fails the erisc build with "section
'.text' is not within region 'ERISC_APP_KERNEL_CODE'"):

    eager-1              1689.19s   acceptance 3.000
    eager-2                 2.80s   acceptance 3.000
    capture                 4.27s
    traced-1                0.48s   acceptance 3.000
    traced-2                0.52s   acceptance 3.000
    traced prompt len 16  137.72s   acceptance 3.000
    traced prompt len 24  282.75s   acceptance 1.500
    traced prompt len 32  141.36s   acceptance 3.000
    traced prompt len 48    8.68s   acceptance 3.000
    traced prompt len 64      --    process DIED, silently

Two things that run counter to how this was being read before:

1. WATCHER MAKES THE FIRST GENERATION LOOK LIKE A HANG AND IT IS NOT ONE. -DWATCHER_ENABLED changes
   the kernel build hash, so the whole 27B kernel set recompiles: 28 minutes of eager-1 with python
   at 116% CPU and hundreds of kernel-cache files appearing per minute. Every new prefill shape in
   the length sweep pays the same toll again (hence 137s / 282s / 141s, then 8.68s once warm).
   Check `find $TT_METAL_CACHE/kernels -newermt '-90 seconds' | wc -l` before calling anything a
   hang under Watcher.

2. THE FAILURE AT len 64 IS A SILENT PROCESS DEATH, NOT A HANG. No traceback, no pytest summary,
   no watcher trip, no readable OOM record -- the process simply stopped between one log line and
   the next. That is a different fault from "blocked forever in ttnn.slice", and it means the
   Python-level hang location recorded earlier (apply_partial_rope_prefill, rope_tp.py:730) was
   probably just where the host happened to be, not where the fault is. Dispatch is asynchronous
   and that op's shapes are CONSTANT across steps (ctx_pad 16 + block 16 = 32 rows), so it cannot
   itself be the thing that degrades with generation count.

Also note the Watcher poll cadence in the log: devices 0-4 answer within milliseconds of each
other, devices 5, 6 and 7 take ~33s EACH, every round, throughout. Unexplained, and the two
ETH-heartbeat failures this work hit both named a single ASIC.
"""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.drafters import TtDrafter
from models.demos.blackhole.qwen36.reference.dflash.generate import dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import (
    DFlashDrafterConfig,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 250_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_which_generation_hangs(mesh_device, device_params, reset_seeds, ensure_gc):
    del device_params
    from transformers import AutoTokenizer

    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=4096)
    model.allocate_kv_caches([64, model.args.n_local_kv_heads, 64, model.args.head_dim], ttnn.bfloat16, batch_size=1)
    pt = torch.arange(64, dtype=torch.int32).unsqueeze(0)
    target = TtTarget(model, cfg.target_layer_ids, pt, device_taps=True)
    drafter = TtDrafter(TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path), tt_ccl=model.tt_ccl), target)
    tok = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tok("The capital of France is", return_tensors="pt").input_ids

    def gen(tag):
        t0 = time.perf_counter()
        logger.info(f">>>>> {tag}: START")
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=4, return_stats=True)
        logger.info(
            f">>>>> {tag}: DONE in {time.perf_counter() - t0:.2f}s, " f"acceptance {stats.mean_acceptance_length:.3f}"
        )
        return stats

    gen("eager-1")
    gen("eager-2")
    logger.info(">>>>> CAPTURING TRACE")
    target.enable_traced_verify()
    logger.info(">>>>> CAPTURE DONE")
    gen("traced-1")
    gen("traced-2")

    # Generation COUNT is not the trigger (traced-1 and traced-2 both complete in <0.5s). The known
    # hangs used 64- and 128-token prompts. capture_verify_trace warms and captures at
    # valid_len = min(16, bucket-1) = 16; a longer prompt replays that same trace at a much larger
    # valid_len. Bisect prompt length under the trace to find where it stops.
    long_text = (
        "The history of computing hardware spans more than a century, beginning with mechanical "
        "calculators and progressing through relays, vacuum tubes, discrete transistors, and finally "
        "integrated circuits. Each transition reduced the cost of a single logical operation by orders "
        "of magnitude, and each one was driven less by a single invention than by the accumulation of "
        "manufacturing technique. The earliest machines were built one at a time by hand, and their "
        "designers could describe every wire; modern processors contain billions of devices that no "
        "individual has examined."
    )
    long_ids = tok(long_text, return_tensors="pt", add_special_tokens=False).input_ids
    for plen in (16, 24, 32, 48, 64):
        if long_ids.shape[1] < plen:
            logger.info(f">>>>> len {plen}: SKIPPED, text too short")
            continue
        t0 = time.perf_counter()
        logger.info(f">>>>> traced prompt len {plen}: START")
        st = dflash_generate(drafter, target, long_ids[:, :plen], max_new_tokens=4, return_stats=True)
        logger.info(
            f">>>>> traced prompt len {plen}: DONE in {time.perf_counter() - t0:.2f}s, "
            f"acceptance {st.mean_acceptance_length:.3f}"
        )

    # Generation LENGTH is the last untested variable: every hang this session used 100/48/24 new
    # tokens, every completion used <= 48, and this file has been using 4. Match the demo exactly.
    for ntok in (24, 48, 100):
        t0 = time.perf_counter()
        logger.info(f">>>>> len 64 x {ntok} new tokens: START")
        st = dflash_generate(drafter, target, long_ids[:, :64], max_new_tokens=ntok, return_stats=True)
        logger.info(
            f">>>>> len 64 x {ntok} new tokens: DONE in {time.perf_counter() - t0:.2f}s, "
            f"acceptance {st.mean_acceptance_length:.3f}"
        )

    logger.info(">>>>> ALL GENERATIONS COMPLETED -- no hang")
    print("\n>>> no hang across 2 eager + 2 traced + prompt-length sweep\n")
