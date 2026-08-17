# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where does the device lose acceptance -- in the drafter port, or in the target port?

On the same prompt at the same length, the CPU oracle accepts **4.41** tokens per
target forward and this port accepts **2.72**.  That 1.6x is worth more than every
remaining perf item combined, and three plausible explanations have already been
measured and *rejected*:

* the LM head's tanh softcap saturating in bf16 (2.72 -> 2.78, no change),
* drafter weight dtype BFP8 vs BF16 (2.72 -> 2.78, and 4x slower),
* drafter math fidelity (LoFi -> HiFi4 + fp32 accumulation, no change).

So the remaining candidates are structural, and they split cleanly in two.  A drafted
token is accepted when ``drafter_argmax == target_argmax``, so acceptance can fall
either because the **drafter** predicts differently from the real drafter, or because
the **target** it is being scored against differs from the real target -- the drafter
is faithful to the *published* model, so wherever the port's argmax deviates from the
real one, a correct prediction is marked wrong.

This runs the real DFlash loop on device and, at each iteration, drafts the block
**twice** from byte-identical inputs -- once with the TTNN drafter, once with the HF
drafter on CPU -- scores both against the *same* device target argmax, and reports:

* ``agreement``: how often the two drafters propose the same token, which is drafter
  port fidelity expressed in the only unit that matters here;
* ``matches_device`` vs ``matches_hf``: accepted tokens per block for each.  If
  ``matches_hf`` is also ~1.8 then the drafter port is exonerated and the target port
  is what caps acceptance; if it is ~3.4 the drafter port is the whole gap.

Both drafters get the same context, anchor and positions, so nothing else varies.

Usage::

    python -m models.autoports.meta_models_muse_glimmer_30b.tests.dflash_acceptance_probe \
        --iterations 12
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference_dflash as R
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_accept import accept_block
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_drafter import DFlashDrafter
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_runner import DFlashRunner
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    build_generator,
    close_generator_mesh,
    open_generator_mesh,
)

PROMPT = "Write a Python function that merges two sorted lists."


class ProbeRunner(DFlashRunner):
    """``DFlashRunner`` that also drafts each block with the HF drafter for comparison."""

    def __init__(self, *args, hf_drafter, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hf_drafter = hf_drafter
        self.records: list[dict] = []

    def _hf_candidates(self, context_host: torch.Tensor, noise_embeds: torch.Tensor) -> list[int]:
        """HF drafter on the same context, scored through the *device* LM head.

        Both sides go through the device LM head and both are handed the **same** noise
        embeddings -- read back from the device tensor rather than looked up again on
        host -- so neither the embedding lookup nor the head is part of the comparison.
        What is left is the drafter body.
        """
        from transformers.cache_utils import DFlashCache

        block = self.config.block_size
        context_len = int(context_host.shape[1])

        cache = DFlashCache(config=self.hf_drafter.config)
        cache.set_previous_accepted_tokens(context_len)
        attention_mask = torch.ones(1, context_len + block, dtype=torch.long)
        with torch.no_grad():
            out = self.hf_drafter(
                noise_embeds=noise_embeds.to(torch.bfloat16),
                context_hidden_states=context_host.to(torch.bfloat16),
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
        hidden = out.last_hidden_state.float()

        tt_hidden = ttnn.from_torch(
            hidden.reshape(1, 1, *hidden.shape[-2:]).to(torch.bfloat16),
            device=self.model.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.mesh_device),
        )
        ids = self._candidate_ids(tt_hidden)
        ttnn.deallocate(tt_hidden)
        return ids

    @staticmethod
    def _noise_to_host(noise: ttnn.Tensor, mesh_device) -> torch.Tensor:
        host = ttnn.to_torch(noise, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]
        return host.reshape(1, host.shape[-2], host.shape[-1]).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--drafter-dtype", default="bfloat8_b", choices=["bfloat8_b", "bfloat16"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mesh = open_generator_mesh()
    try:
        gen = build_generator(".", mesh, max_batch_size=1, max_seq_len=args.max_seq_len)
        tok = gen.tokenizer
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = list(tok(text)["input_ids"])

        drafter = DFlashDrafter.from_state_dict(
            R.draft_state_dict(),
            hf_config=R.draft_config(),
            mesh_device=mesh,
            weight_dtype=getattr(ttnn, args.drafter_dtype),
            activation_dtype=ttnn.bfloat16,
        )
        logger.info("loading the HF drafter on CPU (bf16) ...")
        hf_drafter = R.reference_model(dtype=torch.bfloat16)
        hf_drafter.eval()

        runner = ProbeRunner(gen, drafter, hf_drafter=hf_drafter)

        records = _run(runner, prompt_ids, args.iterations)

        agree = [r["agreement"] for r in records]
        md = [r["matches_device"] for r in records]
        mh = [r["matches_hf"] for r in records]
        print("\n" + "=" * 78)
        print(f"blocks compared                     : {len(records)}")
        print(f"candidate agreement device vs HF    : {statistics.mean(agree):.3f}  (per position, of 15)")
        print(f"accepted per block, device drafter  : {statistics.mean(md):.2f}")
        print(f"accepted per block, HF drafter      : {statistics.mean(mh):.2f}")
        print("-" * 78)
        if statistics.mean(mh) > statistics.mean(md) + 0.5:
            print("VERDICT: the DRAFTER PORT is the gap -- HF candidates score materially better")
            print("against the very same device target, so the target port is not what limits it.")
        else:
            print("VERDICT: the drafter port is EXONERATED -- HF's own candidates score no better")
            print("against this target, so what caps acceptance is the TARGET port's argmax")
            print("differing from the real model the drafter was trained against.")
        print("=" * 78)

        out = Path(args.out) if args.out else Path(__file__).with_name("dflash_acceptance_probe.json")
        out.write_text(
            json.dumps(
                {
                    "blocks": len(records),
                    "mean_agreement": statistics.mean(agree),
                    "mean_matches_device": statistics.mean(md),
                    "mean_matches_hf": statistics.mean(mh),
                    "records": records,
                },
                indent=2,
            )
        )
        print(f"wrote {out}")
    finally:
        close_generator_mesh(mesh)


def _run(runner: ProbeRunner, prompt_ids, iterations: int) -> list[dict]:
    """A cut-down DFlash loop that drafts twice per block."""
    model = runner.model
    block = runner.config.block_size
    eos = tuple(runner.generator._eos_ids)

    runner.generator._invalidate_traces_if_cache_moved()
    runner.generator._allocate_device_inputs()
    table = runner.generator._coerce_page_table(None)
    slot_row = model.page_table_row(table, 0)
    tt_page_table = model.page_table_row_to_device(slot_row)

    model.arm_hidden_state_taps(runner._tap_layers())
    tt_tokens, _ = model.prefill_tokens_to_device(list(prompt_ids))
    embedded = model.embed_prefill(tt_tokens)
    ttnn.deallocate(tt_tokens)
    hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=0, start_pos=0)
    prompt_len = len(prompt_ids)
    context_host = runner._taps_to_host(prompt_len)
    logits = model.prefill_logits(hidden, last_token_index=prompt_len - 1, apply_softcap=not runner.uncapped_argmax)
    ttnn.deallocate(hidden)
    anchor = runner._argmax_rows(logits, model.row_within_tile(prompt_len - 1) + 1)[
        model.row_within_tile(prompt_len - 1)
    ]
    ttnn.deallocate(logits)

    produced = [anchor]
    anchor_pos = prompt_len
    accumulated = context_host
    records: list[dict] = []

    for _ in range(iterations):
        if produced[-1] in eos:
            break
        valid = int(accumulated.shape[1])
        from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_drafter import context_bucket

        width = context_bucket(valid)
        tt_context = runner._upload_context(accumulated, pad_to=width)
        noise = runner._noise_embeds(produced[-1])
        noise_host = runner._noise_to_host(noise, model.mesh_device)
        drafter_out = runner.drafter.forward_padded(noise, tt_context, context_valid=valid, noise_start=anchor_pos)
        ttnn.deallocate(tt_context)
        device_candidates = runner._candidate_ids(drafter_out)
        ttnn.deallocate(drafter_out)
        hf_candidates = runner._hf_candidates(accumulated, noise_host)

        page_block = int(model.config.page_block_size)
        aligned_start = (anchor_pos // page_block) * page_block
        lead = anchor_pos - aligned_start
        full_sequence = list(prompt_ids) + produced
        verify_ids = full_sequence[aligned_start:anchor_pos] + [produced[-1]] + device_candidates

        model.arm_hidden_state_taps(runner._tap_layers())
        tt_tokens, _ = model.prefill_tokens_to_device(verify_ids)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        model.release_sliding_tails()
        hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=0, start_pos=aligned_start)
        rows = model.prefill_all_logits(hidden, prompt_len=len(verify_ids), apply_softcap=not runner.uncapped_argmax)
        all_argmax: list[int] = []
        for tile_index, row in enumerate(rows):
            remaining = len(verify_ids) - tile_index * 32
            all_argmax.extend(runner._argmax_rows(row, min(32, remaining)))
            ttnn.deallocate(row)
        target_argmax = all_argmax[lead : lead + block]

        result = accept_block(device_candidates, target_argmax, eos_token_ids=eos, max_new_tokens=block)
        # Score HF's candidates against the SAME target argmax. Only the leading run
        # counts, exactly as accept_block does, so this is comparable.
        matches_hf = 0
        for cand, want in zip(hf_candidates, target_argmax):
            if cand != want:
                break
            matches_hf += 1
        agreement = sum(1 for a, b in zip(device_candidates, hf_candidates) if a == b) / max(1, len(hf_candidates))

        records.append(
            {
                "anchor_pos": anchor_pos,
                "agreement": agreement,
                "matches_device": result.n_matches,
                "matches_hf": matches_hf,
            }
        )
        logger.info(f"pos {anchor_pos}: agreement {agreement:.2f}  device {result.n_matches}  hf {matches_hf}")

        produced.extend(result.tokens)
        num = result.n_matches + 1
        accumulated = torch.cat([accumulated, runner._taps_to_host(num, offset=lead)], dim=1)
        ttnn.deallocate(hidden)
        anchor_pos += result.n_committed

    model.arm_hidden_state_taps(None)
    model.release_sliding_tails()
    ttnn.deallocate(tt_page_table)
    return records


if __name__ == "__main__":
    main()
