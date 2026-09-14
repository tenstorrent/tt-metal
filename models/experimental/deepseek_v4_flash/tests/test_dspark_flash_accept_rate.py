# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DSpark accept rate against full ttnn DeepSeek-V4-Flash on a real chat prompt.

Prefills the demo prompt on the 32-chip TP4 pipeline, then for each generated
token reads the packed layer-40/41/42 residuals off the MTP D2D socket and runs
the **checkpoint** ``mtp.*`` stack (MLA + 256-expert MoE on the idle submesh)
plus Markov residual. A draft token is accepted when it equals Flash's greedy
argmax at that position.

Requires idle chips (the MTP recv submesh). Skip on 8-chip meshes.

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache DEEPSEEK_V4_DSPARK_ACCEPT_TOKENS=32 \\
      pytest -s models/experimental/deepseek_v4_flash/tests/test_dspark_flash_accept_rate.py
"""

from __future__ import annotations

import contextlib
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.dspark import (
    load_flash_dspark,
    speculative_accept_lengths,
    speculative_accept_rate,
)
from models.experimental.deepseek_v4_flash.tests.test_full_model_decode_demo import (
    _DEFAULT_MODEL_DIR,
    _DEFAULT_TEXT,
    _build_and_prefill,
    _checkpoint_available,
)
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "num_command_queues": 2}],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "mesh_device,tp_size",
    [pytest.param((8, 4), 4, id="tp4_32chip")],
    indirect=["mesh_device"],
)
def test_dspark_flash_accept_rate_real_prompt(mesh_device, reset_seeds, tp_size: int) -> None:
    n_check = int(os.environ.get("DEEPSEEK_V4_DSPARK_ACCEPT_TOKENS", "32"))
    os.environ["DEEPSEEK_V4_MAX_NEW_TOKENS"] = str(n_check)
    # The reference drafter below owns the checkpoint MTP stack. Do not also
    # construct the diagnostic causal S=1 ttnn MTP stack on the idle row.
    os.environ["DEEPSEEK_V4_LOAD_MTP"] = "0"

    with contextlib.ExitStack() as prefetcher:
        state = _build_and_prefill(mesh_device, _DEFAULT_TEXT, prefetcher, tp_size=tp_size)
        model = state["model"]
        tokenizer = state["tokenizer"]
        prompt_ids = state["prompt_ids"]
        real_len = state["real_len"]
        start_pos = state["start_pos"]
        max_seq = state["max_seq"]
        traced = state["traced"]
        next_id = state["next_id"]
        eos_id = state["eos_id"]
        rope = state["rope"]

        if model.mtp_submesh is None or model._mtp_hiddens is None:
            pytest.skip("MTP D2D tap needs idle chips and layers 40-42 (32-chip TP4)")
        if not traced:
            pytest.skip("accept-rate test uses traced decode")

        loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
        logger.info("Loading checkpoint-faithful PyTorch DSpark reference")
        drafter = load_flash_dspark(loader, dequantize_weight).eval()
        gamma = 5

        n_ahead = min(n_check, state["max_new_tokens"], max(0, max_seq - (start_pos + real_len)))
        gen_positions = [start_pos + real_len + i for i in range(n_ahead)]

        records: list[tuple[int, torch.Tensor, int, int]] = []

        def _record(anchor: int, gold_next: int, pos: int) -> None:
            pack = model.read_mtp_hiddens()
            records.append((anchor, pack, gold_next, pos))

        last_prompt_pos = start_pos + real_len - 1
        _record(prompt_ids[-1], next_id, last_prompt_pos)
        token = next_id
        generated = [token]
        for pos in gen_positions:
            logits = model.decode_traced(token, pos).reshape(1, -1).float()
            gold = int(logits[0].argmax().item())
            _record(token, gold, pos)
            generated.append(gold)
            token = gold
            logger.info(f"pos {pos}: gold {gold} {tokenizer.decode([gold])!r}")
            if gold == eos_id:
                break

        drafts = []
        golds = []
        first_hits = 0

        for i, (anchor, _pack, gold, _pos) in enumerate(records):
            # Each record contributes one target position and contains three
            # layer features. Keep all committed features as DSpark context.
            context = torch.stack([p[:, :, 0, :] for _, p, _, _ in records[: i + 1]], dim=1)
            out = drafter(
                context,
                torch.tensor([anchor], dtype=torch.long),
                absolute_position=0,
                greedy=True,
            )
            block = out.draft_ids[0]
            drafts.append(block.cpu())
            golds.append(gold)
            hit = int(block[0]) == gold
            first_hits += int(hit)
            logger.info(
                f"anchor={anchor} gold={gold} draft0={int(block[0])} "
                f"{'HIT' if hit else 'MISS'} block={block.tolist()} "
                f"confidence={out.confidence[0].tolist()}"
            )
        n = len(records)
        first_rate = first_hits / n
        prefix_lens = []
        for i, draft in enumerate(drafts):
            cont = golds[i : i + gamma]
            if not cont:
                continue
            target = torch.tensor(cont, dtype=torch.long)
            prefix_lens.append(
                int(speculative_accept_lengths(draft[: len(cont)].view(1, -1), target.view(1, -1)).item())
            )
        mean_prefix = sum(prefix_lens) / max(len(prefix_lens), 1)

        logger.info(
            f"Flash+checkpoint-MTP accept on {tokenizer.decode(prompt_ids)!r}: "
            f"first-token {first_hits}/{n} = {first_rate:.3f}, "
            f"mean prefix length {mean_prefix:.2f}/{gamma}, "
            f"generated {tokenizer.decode(generated)!r}"
        )
        assert n >= 1
        first_ids = torch.stack([d[:1] for d in drafts])
        gold_t = torch.tensor(golds, dtype=torch.long).view(-1, 1)
        stats = speculative_accept_rate(first_ids, gold_t)
        logger.info(f"speculative_accept_rate first-token stats: {stats}")
        print(f"DSPARK_FLASH_FIRST_TOKEN_ACCEPT={first_rate:.6f} n={n} mean_prefix={mean_prefix:.3f}")
