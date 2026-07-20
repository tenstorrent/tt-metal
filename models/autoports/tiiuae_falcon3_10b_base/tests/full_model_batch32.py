# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full 40-layer active-32, permutation, and inactive-row correctness gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import ttnn
from models.autoports.tiiuae_falcon3_10b_base.tt.generator import _first_device_to_torch, build_generator
from models.common.readiness_check.schema import load_reference

CACHE_EVIDENCE_FIELDS = (
    "first_rank_key_pages",
    "first_rank_value_pages",
    "last_rank_key_pages",
    "last_rank_value_pages",
    "first_rank_key_prefill_final",
    "first_rank_value_prefill_final",
    "first_rank_key_decode_update",
    "first_rank_value_decode_update",
    "last_rank_key_prefill_final",
    "last_rank_value_prefill_final",
    "last_rank_key_decode_update",
    "last_rank_value_decode_update",
)


def _page_rows_are_disjoint(page_table: torch.Tensor, active_batch: int) -> bool:
    assigned: set[int] = set()
    for row in page_table[:active_batch]:
        blocks = {int(value) for value in row.tolist() if int(value) >= 0}
        if assigned.intersection(blocks):
            return False
        assigned.update(blocks)
    return True


def _mutual_topk_rows(left: torch.Tensor, right: torch.Tensor, k: int) -> torch.Tensor:
    left_top = left.topk(k, dim=-1).indices
    right_top = right.topk(k, dim=-1).indices
    return (right_top == left_top[:, :1]).any(dim=-1) & (left_top == right_top[:, :1]).any(dim=-1)


def _row_pcc(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_centered = left - left.mean(dim=-1, keepdim=True)
    right_centered = right - right.mean(dim=-1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=-1)
    denominator = torch.sqrt(left_centered.square().sum(dim=-1) * right_centered.square().sum(dim=-1))
    return numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)


def _trace_ids(generator) -> tuple[object, object]:
    return generator._trace_model_id, generator._trace_sampling_id


def _logical_cache_pages(
    cache: torch.Tensor,
    page_table: torch.Tensor,
    prompt_lens: list[int],
    block_size: int,
) -> torch.Tensor:
    """Gather every page read by SDPA, independent of physical mapping.

    Decode writes position ``prompt_len``.  Including that page is essential
    for the length-64 row, whose decode update crosses from logical page 1 to
    logical page 2.  Dynamic SDPA rounds that three-page read to four pages, so
    the causally masked tail page is evidence too.
    """

    def rounded_pages(token_count: int) -> int:
        live_pages = math.ceil(token_count / block_size)
        if live_pages <= 8:
            return 1 << (live_pages - 1).bit_length()
        return math.ceil(live_pages / 8) * 8

    max_pages = max(rounded_pages(length + 1) for length in prompt_lens)
    logical = torch.zeros(
        (len(prompt_lens), max_pages, *cache.shape[1:]),
        dtype=cache.dtype,
    )
    for user, length in enumerate(prompt_lens):
        read_pages = rounded_pages(length + 1)
        physical_pages = page_table[user, :read_pages].to(torch.long)
        if bool(torch.any(physical_pages < 0)):
            raise RuntimeError(f"user {user} is missing a rounded SDPA cache page")
        logical[user, :read_pages] = cache[physical_pages]
    return logical


def _cache_token_rows(
    cache: torch.Tensor,
    page_table: torch.Tensor,
    positions: list[int],
    block_size: int,
) -> torch.Tensor:
    rows = []
    for user, position in enumerate(positions):
        physical_page = int(page_table[user, position // block_size])
        if physical_page < 0:
            raise RuntimeError(f"user {user} is missing the cache page for position {position}")
        rows.append(cache[physical_page, :, position % block_size, :])
    return torch.stack(rows)


def _persistent_pool_ids(generator) -> tuple[int, ...]:
    return (id(generator._decode_trace_input_pool),) + tuple(
        id(tensor) for tensor in generator._decode_trace_input_pool
    )


def _remap_physical_pages(page_table: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Move every assigned physical page while preserving the logical row map."""
    if num_blocks < 2:
        raise ValueError("physical-page remapping requires at least two cache blocks")
    remapped = page_table.clone()
    assigned = remapped >= 0
    offset = num_blocks // 2
    remapped[assigned] = (remapped[assigned] + offset) % num_blocks
    if torch.equal(remapped, page_table):
        raise RuntimeError("physical-page remapping did not change the page table")
    if torch.unique(remapped[assigned]).numel() != torch.unique(page_table[assigned]).numel():
        raise RuntimeError("physical-page remapping introduced an alias")
    return remapped


def _run_low_level(generator, prompt_rows: list[list[int]], *, page_table: torch.Tensor | None = None):
    prompt_lens = [len(row) for row in prompt_rows]
    active_batch = len(prompt_rows)
    tokens = torch.zeros((active_batch, max(prompt_lens)), dtype=torch.long)
    for slot, row in enumerate(prompt_rows):
        tokens[slot, : len(row)] = torch.tensor(row, dtype=torch.long)
    if page_table is None:
        page_table = generator._make_page_table([length + 2 for length in prompt_lens])
    else:
        page_table = page_table.detach().cpu().to(torch.int32).clone()
        expected_shape = (generator.batch, generator.pages_per_user)
        if tuple(page_table.shape) != expected_shape:
            raise ValueError(f"explicit page table must be {expected_shape}, got {tuple(page_table.shape)}")
    kv_cache = generator._ensure_kv_cache()
    generator.set_sampling_params(active_batch=active_batch)
    prefill_sampled = generator.prefill_forward(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        sampling_mode="device",
    )
    prefill_tokens = _first_device_to_torch(prefill_sampled).reshape(-1)[:active_batch].to(torch.long)
    decode_sampled = generator.decode_forward(
        prefill_tokens,
        torch.tensor(prompt_lens),
        page_table=page_table,
        kv_cache=kv_cache,
        sampling_mode="device",
        enable_trace=True,
        active_batch=active_batch,
    )
    ttnn.synchronize_device(generator.mesh_device)
    decode_tokens = _first_device_to_torch(decode_sampled).reshape(-1)[:active_batch].to(torch.long)
    host_logits = torch.cat(
        [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(generator._trace_logits)],
        dim=-1,
    )[..., : generator.model.vocab_size]
    host_logits = host_logits[0, 0, :active_batch].clone()
    host_argmax = host_logits.argmax(dim=-1)
    positions = _first_device_to_torch(generator._trace_inputs[1]).reshape(-1).to(torch.int32)
    first_key = ttnn.to_torch(ttnn.get_device_tensors(kv_cache[0][0])[0]).float()
    first_value = ttnn.to_torch(ttnn.get_device_tensors(kv_cache[0][1])[0]).float()
    last_key = ttnn.to_torch(ttnn.get_device_tensors(kv_cache[-1][0])[0]).float()
    last_value = ttnn.to_torch(ttnn.get_device_tensors(kv_cache[-1][1])[0]).float()
    block_size = generator.model.page_block_size
    return {
        "prompt_lens": prompt_lens,
        "page_table": page_table,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "host_logits": host_logits,
        "host_argmax": host_argmax,
        "positions": positions,
        "first_rank_key_pages": _logical_cache_pages(first_key, page_table, prompt_lens, block_size),
        "first_rank_value_pages": _logical_cache_pages(first_value, page_table, prompt_lens, block_size),
        "last_rank_key_pages": _logical_cache_pages(last_key, page_table, prompt_lens, block_size),
        "last_rank_value_pages": _logical_cache_pages(last_value, page_table, prompt_lens, block_size),
        "first_rank_key_prefill_final": _cache_token_rows(
            first_key, page_table, [length - 1 for length in prompt_lens], block_size
        ),
        "first_rank_value_prefill_final": _cache_token_rows(
            first_value, page_table, [length - 1 for length in prompt_lens], block_size
        ),
        "first_rank_key_decode_update": _cache_token_rows(first_key, page_table, prompt_lens, block_size),
        "first_rank_value_decode_update": _cache_token_rows(first_value, page_table, prompt_lens, block_size),
        "last_rank_key_prefill_final": _cache_token_rows(
            last_key, page_table, [length - 1 for length in prompt_lens], block_size
        ),
        "last_rank_value_prefill_final": _cache_token_rows(
            last_value, page_table, [length - 1 for length in prompt_lens], block_size
        ),
        "last_rank_key_decode_update": _cache_token_rows(last_key, page_table, prompt_lens, block_size),
        "last_rank_value_decode_update": _cache_token_rows(last_value, page_table, prompt_lens, block_size),
    }


def collect(
    model_dir: Path,
    reference_path: Path,
    output: Path,
    weight_cache_path: str,
    override_num_layers: int | None = None,
    slot_probe_only: bool = False,
) -> dict:
    reference = load_reference(reference_path)
    base_prompt = reference.entries[0].prompt_tokens[0].tolist()
    prompt_rows = [base_prompt[slot : slot + 33 + slot] for slot in range(32)]
    if [len(row) for row in prompt_rows] != list(range(33, 65)):
        raise ValueError("reference prompt is too short for the active-32 control")

    mesh = None
    generator = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=512_000_000)
        generator = build_generator(
            model_dir,
            mesh,
            max_batch_size=32,
            max_context_len=256,
            override_num_layers=override_num_layers,
            weight_cache_path=weight_cache_path,
        )

        persistent_pool_ids = _persistent_pool_ids(generator)
        run_lifecycle: dict[str, dict[str, object]] = {}
        reset_lifecycle: dict[str, dict[str, object]] = {}

        def run_case(name: str, rows: list[list[int]], *, page_table=None):
            stats_before = dict(generator.trace_stats)
            trace_ids_before = _trace_ids(generator)
            result = _run_low_level(generator, rows, page_table=page_table)
            stats_after = dict(generator.trace_stats)
            trace_ids_after = _trace_ids(generator)
            run_lifecycle[name] = {
                "release_delta": stats_after["releases"] - stats_before["releases"],
                "capture_delta": stats_after["captures"] - stats_before["captures"],
                "page_table_host_copy_delta": stats_after["page_table_host_copies"]
                - stats_before["page_table_host_copies"],
                "trace_installed_after": all(trace_id is not None for trace_id in trace_ids_after),
                "trace_ids_changed": trace_ids_after != trace_ids_before,
                "persistent_pool_ids_stable": _persistent_pool_ids(generator) == persistent_pool_ids,
            }
            return result

        def reset_case(name: str) -> None:
            stats_before = dict(generator.trace_stats)
            trace_ids_before = _trace_ids(generator)
            generator.reset()
            stats_after = dict(generator.trace_stats)
            reset_lifecycle[name] = {
                "trace_ids_released": _trace_ids(generator) == (None, None),
                "trace_ids_were_installed": all(trace_id is not None for trace_id in trace_ids_before),
                "release_delta": stats_after["releases"] - stats_before["releases"],
                "capture_delta": stats_after["captures"] - stats_before["captures"],
                "persistent_pool_ids_stable": _persistent_pool_ids(generator) == persistent_pool_ids,
            }

        # A: canonical slots and canonical physical pages.
        forward = run_case("forward", prompt_rows)
        reset_case("after_forward")
        reset_released_trace = bool(reset_lifecycle["after_forward"]["trace_ids_released"])
        reset_release_delta = int(reset_lifecycle["after_forward"]["release_delta"])

        # A': exact repeat.  This separates nondeterminism/reset contamination
        # from either of the slot/page metamorphic axes below.
        repeat = run_case("repeat", prompt_rows, page_table=forward["page_table"])
        repeat_prefill_match = torch.equal(repeat["prefill_tokens"], forward["prefill_tokens"])
        repeat_decode_match = torch.equal(repeat["decode_tokens"], forward["decode_tokens"])
        repeat_logits_match = torch.equal(repeat["host_logits"], forward["host_logits"])
        repeat_logits_delta = float((repeat["host_logits"] - forward["host_logits"]).abs().max())
        repeat_top5_rows = _mutual_topk_rows(forward["host_logits"], repeat["host_logits"], 5)
        repeat_pcc_rows = _row_pcc(forward["host_logits"], repeat["host_logits"])
        repeat_logit_row_deltas = (repeat["host_logits"] - forward["host_logits"]).abs().amax(dim=-1)
        repeat_positions_match = torch.equal(repeat["positions"], forward["positions"])
        repeat_cache_matches = {name: torch.equal(repeat[name], forward[name]) for name in CACHE_EVIDENCE_FIELDS}

        # B: reverse fixed slots but carry each logical user's physical page
        # row with it.  After flipping outputs back to logical-user order this
        # is state-equivalent to A at tokens, positions, and cache addresses.
        reset_case("after_repeat")
        reverse_page_table = forward["page_table"].flip(0).contiguous()
        reverse = run_case(
            "reverse_preserved_pages",
            list(reversed(prompt_rows)),
            page_table=reverse_page_table,
        )
        reverse_page_copy_delta = int(run_lifecycle["reverse_preserved_pages"]["page_table_host_copy_delta"])
        reverse_reused_trace = bool(
            run_lifecycle["reverse_preserved_pages"]["release_delta"] == 0
            and run_lifecycle["reverse_preserved_pages"]["capture_delta"] == 0
        )
        reverse_recaptured_trace = bool(
            run_lifecycle["reverse_preserved_pages"]["release_delta"] == 0
            and run_lifecycle["reverse_preserved_pages"]["capture_delta"] == 1
            and run_lifecycle["reverse_preserved_pages"]["trace_installed_after"]
        )
        prefill_permutation_match = torch.equal(reverse["prefill_tokens"].flip(0), forward["prefill_tokens"])
        decode_permutation_match = torch.equal(reverse["decode_tokens"].flip(0), forward["decode_tokens"])
        reverse_positions_in_forward_order = reverse["positions"].flip(0)
        reverse_positions_match = torch.equal(reverse_positions_in_forward_order, forward["positions"])
        reverse_logical_page_rows_preserved = torch.equal(reverse["page_table"].flip(0), forward["page_table"])
        reverse_logits_in_forward_order = reverse["host_logits"].flip(0)
        decode_logits_permutation_match = torch.equal(reverse_logits_in_forward_order, forward["host_logits"])
        decode_logits_permutation_delta = float((reverse_logits_in_forward_order - forward["host_logits"]).abs().max())
        reverse_key_pages = reverse["first_rank_key_pages"].flip(0)
        reverse_value_pages = reverse["first_rank_value_pages"].flip(0)
        key_cache_permutation_match = torch.equal(reverse_key_pages, forward["first_rank_key_pages"])
        value_cache_permutation_match = torch.equal(reverse_value_pages, forward["first_rank_value_pages"])
        key_cache_permutation_delta = float((reverse_key_pages - forward["first_rank_key_pages"]).abs().max())
        value_cache_permutation_delta = float((reverse_value_pages - forward["first_rank_value_pages"]).abs().max())
        reverse_last_key_pages = reverse["last_rank_key_pages"].flip(0)
        reverse_last_value_pages = reverse["last_rank_value_pages"].flip(0)
        last_key_cache_permutation_match = torch.equal(reverse_last_key_pages, forward["last_rank_key_pages"])
        last_value_cache_permutation_match = torch.equal(reverse_last_value_pages, forward["last_rank_value_pages"])
        last_key_cache_permutation_delta = float((reverse_last_key_pages - forward["last_rank_key_pages"]).abs().max())
        last_value_cache_permutation_delta = float(
            (reverse_last_value_pages - forward["last_rank_value_pages"]).abs().max()
        )
        decode_permutation_top5_rows = _mutual_topk_rows(forward["host_logits"], reverse_logits_in_forward_order, 5)
        decode_permutation_pcc = _row_pcc(forward["host_logits"], reverse_logits_in_forward_order)
        decode_permutation_row_deltas = (reverse_logits_in_forward_order - forward["host_logits"]).abs().amax(dim=-1)
        reverse_cache_matches = {
            name: torch.equal(reverse[name].flip(0), forward[name]) for name in CACHE_EVIDENCE_FIELDS
        }

        if slot_probe_only:
            # Fast layer-depth bisection surface: A/A'/B only.  The complete
            # stage gate below still runs the full slot x physical-page matrix,
            # inactive-row check, and representative one-active-user controls.
            result = {
                "probe": "state-equivalent fixed-slot permutation",
                "mesh": "4x Blackhole p300c, 1x4 FABRIC_1D_RING, TP4",
                "layers_executed": generator.model.num_layers,
                "fixed_slots": generator.batch,
                "active_slots": 32,
                "prompt_lens": forward["prompt_lens"],
                "page_rows_disjoint": _page_rows_are_disjoint(forward["page_table"], 32),
                "repeat_same_mapping_prefill_tokens_exact": repeat_prefill_match,
                "repeat_same_mapping_decode_tokens_exact": repeat_decode_match,
                "repeat_same_mapping_logits_exact": repeat_logits_match,
                "repeat_same_mapping_logits_max_abs_delta": repeat_logits_delta,
                "repeat_same_mapping_logits_max_abs_delta_rows": repeat_logit_row_deltas.tolist(),
                "repeat_same_mapping_mutual_top5_rows": repeat_top5_rows.tolist(),
                "repeat_same_mapping_mutual_top5_match_all_32": bool(torch.all(repeat_top5_rows)),
                "repeat_same_mapping_pcc_rows": repeat_pcc_rows.tolist(),
                "repeat_same_mapping_positions_exact": repeat_positions_match,
                "repeat_same_mapping_cache_exact": repeat_cache_matches,
                "reverse_logical_page_rows_preserved": reverse_logical_page_rows_preserved,
                "reverse_reused_same_trace": reverse_reused_trace,
                "reverse_released_and_recaptured_trace": reverse_recaptured_trace,
                "reverse_positions_exact_in_logical_order": reverse_positions_match,
                "prefill_permutation_match_all_32": prefill_permutation_match,
                "decode_permutation_match_all_32": decode_permutation_match,
                "decode_logits_permutation_exact_all_32": decode_logits_permutation_match,
                "decode_permutation_mutual_top5_rows": decode_permutation_top5_rows.tolist(),
                "decode_permutation_mutual_top5_match_all_32": bool(torch.all(decode_permutation_top5_rows)),
                "decode_permutation_pcc_rows": decode_permutation_pcc.tolist(),
                "decode_logits_permutation_max_abs_delta_rows": decode_permutation_row_deltas.tolist(),
                "decode_logits_permutation_max_abs_delta": decode_logits_permutation_delta,
                "first_layer_key_cache_permutation_exact": key_cache_permutation_match,
                "first_layer_key_cache_permutation_max_abs_delta": key_cache_permutation_delta,
                "first_layer_value_cache_permutation_exact": value_cache_permutation_match,
                "first_layer_value_cache_permutation_max_abs_delta": value_cache_permutation_delta,
                "last_layer_key_cache_permutation_exact": last_key_cache_permutation_match,
                "last_layer_key_cache_permutation_max_abs_delta": last_key_cache_permutation_delta,
                "last_layer_value_cache_permutation_exact": last_value_cache_permutation_match,
                "last_layer_value_cache_permutation_max_abs_delta": last_value_cache_permutation_delta,
                "reverse_all_live_cache_pages_exact": reverse_cache_matches,
                "forward_split_matches_host_argmax_all_32": torch.equal(
                    forward["decode_tokens"], forward["host_argmax"]
                ),
                "repeat_split_matches_host_argmax_all_32": torch.equal(repeat["decode_tokens"], repeat["host_argmax"]),
                "reverse_split_matches_host_argmax_all_32": torch.equal(
                    reverse["decode_tokens"], reverse["host_argmax"]
                ),
                "persistent_pool_ids": list(persistent_pool_ids),
                "run_lifecycle": run_lifecycle,
                "reset_lifecycle": reset_lifecycle,
                "trace_stats": dict(generator.trace_stats),
            }
            result["passed"] = bool(
                result["layers_executed"] == (40 if override_num_layers is None else override_num_layers)
                and result["fixed_slots"] == 32
                and result["active_slots"] == 32
                and result["page_rows_disjoint"]
                and result["repeat_same_mapping_prefill_tokens_exact"]
                and result["repeat_same_mapping_decode_tokens_exact"]
                and result["repeat_same_mapping_logits_exact"]
                and result["repeat_same_mapping_positions_exact"]
                and all(result["repeat_same_mapping_cache_exact"].values())
                and result["reverse_logical_page_rows_preserved"]
                and not result["reverse_reused_same_trace"]
                and result["reverse_released_and_recaptured_trace"]
                and result["reverse_positions_exact_in_logical_order"]
                and result["prefill_permutation_match_all_32"]
                and result["decode_permutation_match_all_32"]
                and result["decode_logits_permutation_exact_all_32"]
                and all(result["reverse_all_live_cache_pages_exact"].values())
                and result["first_layer_key_cache_permutation_exact"]
                and result["first_layer_value_cache_permutation_exact"]
                and result["last_layer_key_cache_permutation_exact"]
                and result["last_layer_value_cache_permutation_exact"]
                and result["forward_split_matches_host_argmax_all_32"]
                and result["repeat_split_matches_host_argmax_all_32"]
                and result["reverse_split_matches_host_argmax_all_32"]
                and run_lifecycle["forward"]["release_delta"] == 0
                and run_lifecycle["forward"]["capture_delta"] == 1
                and all(
                    lifecycle["release_delta"] == 0
                    and lifecycle["capture_delta"] == 1
                    and lifecycle["trace_installed_after"]
                    and lifecycle["persistent_pool_ids_stable"]
                    for name, lifecycle in run_lifecycle.items()
                    if name != "forward"
                )
                and all(
                    lifecycle["trace_ids_released"]
                    and lifecycle["trace_ids_were_installed"]
                    and lifecycle["release_delta"] == 1
                    and lifecycle["capture_delta"] == 0
                    and lifecycle["persistent_pool_ids_stable"]
                    for lifecycle in reset_lifecycle.values()
                )
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2))
            return result

        # C: retain canonical slots but move every logical page to a different
        # physical block.  This isolates cache-address dependence from slot
        # dependence.  D supplies the remaining factorial interaction cell.
        remapped_page_table = _remap_physical_pages(forward["page_table"], generator.num_blocks)
        reset_case("after_reverse_preserved_pages")
        remapped = run_case("remapped_pages", prompt_rows, page_table=remapped_page_table)
        remapped_prefill_match = torch.equal(remapped["prefill_tokens"], forward["prefill_tokens"])
        remapped_decode_match = torch.equal(remapped["decode_tokens"], forward["decode_tokens"])
        remapped_logits_match = torch.equal(remapped["host_logits"], forward["host_logits"])
        remapped_logits_delta = float((remapped["host_logits"] - forward["host_logits"]).abs().max())
        remapped_top5 = _mutual_topk_rows(forward["host_logits"], remapped["host_logits"], 5)
        remapped_positions_match = torch.equal(remapped["positions"], forward["positions"])
        remapped_cache_matches = {name: torch.equal(remapped[name], forward[name]) for name in CACHE_EVIDENCE_FIELDS}

        reset_case("after_remapped_pages")
        reverse_remapped = run_case(
            "reverse_remapped_pages",
            list(reversed(prompt_rows)),
            page_table=remapped_page_table.flip(0).contiguous(),
        )
        reverse_remapped_logits = reverse_remapped["host_logits"].flip(0)
        reverse_remapped_prefill_match = torch.equal(
            reverse_remapped["prefill_tokens"].flip(0), forward["prefill_tokens"]
        )
        reverse_remapped_decode_match = torch.equal(reverse_remapped["decode_tokens"].flip(0), forward["decode_tokens"])
        reverse_remapped_logits_match = torch.equal(reverse_remapped_logits, forward["host_logits"])
        reverse_remapped_logits_delta = float((reverse_remapped_logits - forward["host_logits"]).abs().max())
        reverse_remapped_positions_match = torch.equal(reverse_remapped["positions"].flip(0), forward["positions"])
        reverse_remapped_top5 = _mutual_topk_rows(forward["host_logits"], reverse_remapped_logits, 5)
        reverse_remapped_cache_matches = {
            name: torch.equal(reverse_remapped[name].flip(0), forward[name]) for name in CACHE_EVIDENCE_FIELDS
        }

        reset_case("after_reverse_remapped_pages")
        first_sixteen_page_table = torch.full_like(forward["page_table"], -1)
        first_sixteen_page_table[:16] = forward["page_table"][:16]
        first_sixteen = run_case("active16", prompt_rows[:16], page_table=first_sixteen_page_table)
        inactive_noninterference_prefill = torch.equal(first_sixteen["prefill_tokens"], forward["prefill_tokens"][:16])
        inactive_noninterference_decode = torch.equal(first_sixteen["decode_tokens"], forward["decode_tokens"][:16])
        inactive_positions = first_sixteen["positions"][16:]

        representative_controls = {}
        previous_case = "active16"
        for user in (0, 15, 31):
            reset_case(f"after_{previous_case}")
            control_page_table = torch.full_like(forward["page_table"], -1)
            control_page_table[0] = forward["page_table"][user]
            case_name = f"active1_user_{user}"
            control = run_case(case_name, [prompt_rows[user]], page_table=control_page_table)
            previous_case = case_name
            representative_controls[str(user)] = {
                "prompt_len": len(prompt_rows[user]),
                "prefill_token": int(control["prefill_tokens"][0]),
                "decode_token": int(control["decode_tokens"][0]),
                "prefill_matches_active32": int(control["prefill_tokens"][0]) == int(forward["prefill_tokens"][user]),
                "decode_matches_active32": int(control["decode_tokens"][0]) == int(forward["decode_tokens"][user]),
                "host_argmax": int(control["host_argmax"][0]),
                "split_matches_host_argmax": int(control["decode_tokens"][0]) == int(control["host_argmax"][0]),
                "logits_max_abs_delta_from_active32": float(
                    (control["host_logits"][0] - forward["host_logits"][user]).abs().max()
                ),
                "mutual_top5_with_active32": bool(
                    _mutual_topk_rows(forward["host_logits"][user : user + 1], control["host_logits"], 5)[0]
                ),
                "logits_pcc_with_active32": float(
                    _row_pcc(forward["host_logits"][user : user + 1], control["host_logits"])[0]
                ),
            }

        result = {
            "mesh": "4x Blackhole p300c, 1x4 FABRIC_1D_RING, TP4",
            "layers_executed": generator.model.num_layers,
            "fixed_slots": generator.batch,
            "active_slots": 32,
            "context_tokens_for_batch_gate": generator.model.max_cache_len,
            "prompt_lens": forward["prompt_lens"],
            "all_prompts_non_aligned_to_128": all(length % 128 for length in forward["prompt_lens"]),
            "page_rows_disjoint": _page_rows_are_disjoint(forward["page_table"], 32),
            "forward_prefill_tokens": forward["prefill_tokens"].tolist(),
            "forward_decode_tokens": forward["decode_tokens"].tolist(),
            "forward_host_argmax_tokens": forward["host_argmax"].tolist(),
            "forward_split_matches_host_argmax_all_32": torch.equal(forward["decode_tokens"], forward["host_argmax"]),
            "repeat_split_matches_host_argmax_all_32": torch.equal(repeat["decode_tokens"], repeat["host_argmax"]),
            "forward_positions": forward["positions"].tolist(),
            "forward_positions_match": forward["positions"].tolist()
            == [length + 1 for length in forward["prompt_lens"]],
            "repeat_same_mapping_prefill_tokens_exact": repeat_prefill_match,
            "repeat_same_mapping_decode_tokens_exact": repeat_decode_match,
            "repeat_same_mapping_logits_exact": repeat_logits_match,
            "repeat_same_mapping_logits_max_abs_delta": repeat_logits_delta,
            "repeat_same_mapping_logits_max_abs_delta_rows": repeat_logit_row_deltas.tolist(),
            "repeat_same_mapping_mutual_top5_rows": repeat_top5_rows.tolist(),
            "repeat_same_mapping_mutual_top5_match_all_32": bool(torch.all(repeat_top5_rows)),
            "repeat_same_mapping_pcc_rows": repeat_pcc_rows.tolist(),
            "repeat_same_mapping_positions_exact": repeat_positions_match,
            "repeat_same_mapping_cache_exact": repeat_cache_matches,
            "reverse_prompt_lens": reverse["prompt_lens"],
            "reverse_page_table_differs": not torch.equal(reverse["page_table"], forward["page_table"]),
            "reverse_logical_page_rows_preserved": reverse_logical_page_rows_preserved,
            "reverse_page_table_host_copy_delta": reverse_page_copy_delta,
            "reverse_reused_same_trace": reverse_reused_trace,
            "reverse_released_and_recaptured_trace": reverse_recaptured_trace,
            "reverse_positions_exact_in_logical_order": reverse_positions_match,
            "prefill_permutation_match_all_32": prefill_permutation_match,
            "decode_permutation_match_all_32": decode_permutation_match,
            "decode_logits_permutation_exact_all_32": decode_logits_permutation_match,
            "decode_permutation_mutual_top5_rows": decode_permutation_top5_rows.tolist(),
            "decode_permutation_mutual_top5_match_all_32": bool(torch.all(decode_permutation_top5_rows)),
            "decode_permutation_pcc_rows": decode_permutation_pcc.tolist(),
            "decode_logits_permutation_max_abs_delta_rows": decode_permutation_row_deltas.tolist(),
            "decode_logits_permutation_max_abs_delta": decode_logits_permutation_delta,
            "decode_logits_permutation_min_pcc": float(decode_permutation_pcc.min()),
            "decode_logits_permutation_mean_pcc": float(decode_permutation_pcc.mean()),
            "reverse_split_matches_host_argmax_all_32": torch.equal(reverse["decode_tokens"], reverse["host_argmax"]),
            "first_layer_key_cache_permutation_exact": key_cache_permutation_match,
            "first_layer_key_cache_permutation_max_abs_delta": key_cache_permutation_delta,
            "first_layer_value_cache_permutation_exact": value_cache_permutation_match,
            "first_layer_value_cache_permutation_max_abs_delta": value_cache_permutation_delta,
            "last_layer_key_cache_permutation_exact": last_key_cache_permutation_match,
            "last_layer_key_cache_permutation_max_abs_delta": last_key_cache_permutation_delta,
            "last_layer_value_cache_permutation_exact": last_value_cache_permutation_match,
            "last_layer_value_cache_permutation_max_abs_delta": last_value_cache_permutation_delta,
            "reverse_all_live_cache_pages_exact": reverse_cache_matches,
            "remapped_page_table_differs": not torch.equal(remapped["page_table"], forward["page_table"]),
            "remapped_page_rows_disjoint": _page_rows_are_disjoint(remapped["page_table"], 32),
            "same_slots_remapped_pages_prefill_tokens_exact": remapped_prefill_match,
            "same_slots_remapped_pages_decode_tokens_exact": remapped_decode_match,
            "same_slots_remapped_pages_logits_exact": remapped_logits_match,
            "same_slots_remapped_pages_logits_max_abs_delta": remapped_logits_delta,
            "same_slots_remapped_pages_mutual_top5_rows": remapped_top5.tolist(),
            "same_slots_remapped_pages_mutual_top5_match_all_32": bool(torch.all(remapped_top5)),
            "same_slots_remapped_pages_positions_exact": remapped_positions_match,
            "same_slots_remapped_pages_split_matches_host_argmax_all_32": torch.equal(
                remapped["decode_tokens"], remapped["host_argmax"]
            ),
            "same_slots_remapped_pages_cache_exact": remapped_cache_matches,
            "reversed_slots_remapped_pages_prefill_tokens_exact": reverse_remapped_prefill_match,
            "reversed_slots_remapped_pages_decode_tokens_exact": reverse_remapped_decode_match,
            "reversed_slots_remapped_pages_logits_exact": reverse_remapped_logits_match,
            "reversed_slots_remapped_pages_logits_max_abs_delta": reverse_remapped_logits_delta,
            "reversed_slots_remapped_pages_positions_exact": reverse_remapped_positions_match,
            "reversed_slots_remapped_pages_split_matches_host_argmax_all_32": torch.equal(
                reverse_remapped["decode_tokens"], reverse_remapped["host_argmax"]
            ),
            "reversed_slots_remapped_pages_mutual_top5_rows": reverse_remapped_top5.tolist(),
            "reversed_slots_remapped_pages_mutual_top5_match_all_32": bool(torch.all(reverse_remapped_top5)),
            "reversed_slots_remapped_pages_cache_exact": reverse_remapped_cache_matches,
            "reset_released_trace": reset_released_trace,
            "reset_release_delta": reset_release_delta,
            "active16_prefill_matches_active32": inactive_noninterference_prefill,
            "active16_decode_matches_active32": inactive_noninterference_decode,
            "active16_inactive_positions_unchanged": bool(torch.all(inactive_positions == -1)),
            "representative_batch1_controls": representative_controls,
            "persistent_pool_ids": list(persistent_pool_ids),
            "run_lifecycle": run_lifecycle,
            "reset_lifecycle": reset_lifecycle,
            "trace_stats": dict(generator.trace_stats),
        }
        result["passed"] = bool(
            result["layers_executed"] == (40 if override_num_layers is None else override_num_layers)
            and result["fixed_slots"] == 32
            and result["active_slots"] == 32
            and result["all_prompts_non_aligned_to_128"]
            and result["page_rows_disjoint"]
            and result["forward_positions_match"]
            and result["repeat_same_mapping_prefill_tokens_exact"]
            and result["repeat_same_mapping_decode_tokens_exact"]
            and result["repeat_same_mapping_logits_exact"]
            and result["repeat_same_mapping_positions_exact"]
            and all(result["repeat_same_mapping_cache_exact"].values())
            and result["reverse_page_table_differs"]
            and result["reverse_logical_page_rows_preserved"]
            and result["reverse_page_table_host_copy_delta"] == 4
            and not result["reverse_reused_same_trace"]
            and result["reverse_released_and_recaptured_trace"]
            and result["reverse_positions_exact_in_logical_order"]
            and result["prefill_permutation_match_all_32"]
            and result["decode_permutation_match_all_32"]
            and result["decode_logits_permutation_exact_all_32"]
            and all(result["reverse_all_live_cache_pages_exact"].values())
            and result["first_layer_key_cache_permutation_exact"]
            and result["first_layer_value_cache_permutation_exact"]
            and result["last_layer_key_cache_permutation_exact"]
            and result["last_layer_value_cache_permutation_exact"]
            and result["forward_split_matches_host_argmax_all_32"]
            and result["repeat_split_matches_host_argmax_all_32"]
            and result["reverse_split_matches_host_argmax_all_32"]
            and result["remapped_page_table_differs"]
            and result["remapped_page_rows_disjoint"]
            and result["same_slots_remapped_pages_prefill_tokens_exact"]
            and result["same_slots_remapped_pages_decode_tokens_exact"]
            and result["same_slots_remapped_pages_logits_exact"]
            and result["same_slots_remapped_pages_positions_exact"]
            and result["same_slots_remapped_pages_split_matches_host_argmax_all_32"]
            and all(result["same_slots_remapped_pages_cache_exact"].values())
            and result["reversed_slots_remapped_pages_prefill_tokens_exact"]
            and result["reversed_slots_remapped_pages_decode_tokens_exact"]
            and result["reversed_slots_remapped_pages_logits_exact"]
            and result["reversed_slots_remapped_pages_positions_exact"]
            and result["reversed_slots_remapped_pages_split_matches_host_argmax_all_32"]
            and all(result["reversed_slots_remapped_pages_cache_exact"].values())
            and result["reset_released_trace"]
            and result["reset_release_delta"] == 1
            and result["active16_prefill_matches_active32"]
            and result["active16_decode_matches_active32"]
            and result["active16_inactive_positions_unchanged"]
            and all(
                control["prefill_matches_active32"]
                and control["mutual_top5_with_active32"]
                and control["split_matches_host_argmax"]
                for control in result["representative_batch1_controls"].values()
            )
            and run_lifecycle["forward"]["release_delta"] == 0
            and run_lifecycle["forward"]["capture_delta"] == 1
            and all(
                lifecycle["release_delta"] == 0
                and lifecycle["capture_delta"] == 1
                and lifecycle["trace_installed_after"]
                and lifecycle["persistent_pool_ids_stable"]
                for name, lifecycle in run_lifecycle.items()
                if name != "forward"
            )
            and all(
                lifecycle["trace_ids_released"]
                and lifecycle["trace_ids_were_installed"]
                and lifecycle["release_delta"] == 1
                and lifecycle["capture_delta"] == 0
                and lifecycle["persistent_pool_ids_stable"]
                for lifecycle in reset_lifecycle.values()
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    finally:
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-cache-path", default="/tmp/falcon3-full-model-cache")
    parser.add_argument("--override-num-layers", type=int)
    parser.add_argument(
        "--slot-probe-only",
        action="store_true",
        help="run only exact repeat plus state-equivalent slot permutation for layer-depth bisection",
    )
    args = parser.parse_args()
    result = collect(
        args.model_dir,
        args.reference,
        args.output,
        args.weight_cache_path,
        override_num_layers=args.override_num_layers,
        slot_probe_only=args.slot_probe_only,
    )
    if not result["passed"]:
        raise SystemExit("full-model active-32 correctness gate failed")


if __name__ == "__main__":
    main()
