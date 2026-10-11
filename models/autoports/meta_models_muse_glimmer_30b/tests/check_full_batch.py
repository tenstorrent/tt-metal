# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-slot, mixed prompt and cache ownership contract, up to batch32."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.check_generate_cache_reuse import check_cache_reset_equivalence
from models.autoports.meta_models_muse_glimmer_30b.tests.check_live_admission import check_live_admission
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator

ROOT = Path(__file__).parents[1]


def main(args):
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    generator = None
    try:
        generator = build_generator(
            ROOT, mesh, batch_size=args.batch, **({"layer_indices": [0, 3]} if args.reduced else {})
        )
        if args.diagnose_eager:

            def watch(method, label):
                def forward(*positional, **keywords):
                    norm_input = None
                    if (
                        label == "_norm"
                        and len(positional) > 1
                        and positional[1] == "post_attention_layernorm"
                        and positional[0].shape[-2] == 64
                    ):
                        norm_input = ttnn.to_torch(ttnn.get_device_tensors(positional[0])[0]).clone()
                    value = method(*positional, **keywords)
                    host = ttnn.to_torch(ttnn.get_device_tensors(value)[0]).float()
                    if norm_input is not None:
                        layer = generator.model.layers[0]
                        weight = ttnn.to_torch(
                            ttnn.get_device_tensors(layer.norm_weights["post_attention_layernorm"])[0]
                        )
                        torch.save(
                            dict(
                                input=norm_input,
                                output=host,
                                weight=weight,
                                epsilon=layer.config.post_norm_eps,
                                config=str(layer.norm_configs["prefill"]["post_attention_layernorm"]),
                            ),
                            args.output.with_suffix(".norm.pt"),
                        )
                    print(
                        "EAGER_BOUNDARY",
                        label,
                        tuple(host.shape),
                        bool(torch.isfinite(host).all()),
                        host.min().item(),
                        host.max().item(),
                        host.flatten()[:8].tolist(),
                        flush=True,
                    )
                    if not torch.isfinite(host).all():
                        print(
                            "NONFINITE_INDICES", label, torch.nonzero(~torch.isfinite(host))[:12].tolist(), flush=True
                        )
                    return value

                return forward

            generator.model.embed = watch(generator.model.embed, "embedding")
            for layer in generator.model.layers:
                layer._prefill_chunk = watch(layer._prefill_chunk, "layer_" + str(layer.layer_idx))
            if args.diagnose_ops:
                for name in ("_norm", "_linear", "_rope", "_add"):
                    setattr(generator.model.layers[0], name, watch(getattr(generator.model.layers[0], name), name))
                ttnn.transformer.chunked_scaled_dot_product_attention = watch(
                    ttnn.transformer.chunked_scaled_dot_product_attention, "sdpa"
                )

            def eager_execute(key):
                if isinstance(key, tuple):
                    if len(key) == 2 and key[1] in generator.BUCKETS:
                        if key[0] == "prefill":
                            return generator._prefill_graph(key[1])
                        if key[0] == "head":
                            return generator._head_graph(key[1])
                    elif len(key) == 3 and key[:2] == ("prefill", 32) and key[2] in generator.APPEND_ALIGNMENTS:
                        return generator._prefill_graph(key[1], key[2])
                elif key == "reset":
                    return generator._reset_cache_graph()
                elif key == "admission_save":
                    return generator._save_admission_graph()
                elif key == "admission_restore":
                    return generator._restore_admission_graph()
                elif key == "decode":
                    return generator._decode_graph()
                elif key == "sample":
                    return generator._sample_graph()
                elif key == "argmax":
                    return generator._sample_graph(force_argmax=True)
                raise ValueError(f"Unknown eager graph key: {key!r}")

            generator._execute = eager_execute
        # Give each fixed slot128 private physical pages. The unused logical
        # entries alias only its own pool; tested positions never exceed130.
        table = torch.stack([torch.arange(4096, dtype=torch.int32) % 128 + slot * 128 for slot in range(args.batch)])
        lengths = [31 + 2 * (slot % 2) for slot in range(args.batch)]
        ids = torch.arange(32, 65)[None].expand(args.batch, -1).clone()
        if args.diagnose_ops:
            lengths, ids = lengths[:2], ids[:2]
        before = dict(generator.counters)
        first = generator.prefill_forward(
            ids, page_table=table, kv_cache=generator.kv_cache, prompt_lens=lengths, sample=False
        )
        comparisons = [
            dict(
                slot=slot,
                expected_slot=slot % 2,
                equal=torch.equal(first[slot], first[slot % 2]),
                max_abs=(first[slot].float() - first[slot % 2].float()).abs().max().item(),
                cosine=torch.nn.functional.cosine_similarity(
                    first[slot].float(), first[slot % 2].float(), dim=-1
                ).item(),
                token=first[slot].argmax().item(),
            )
            for slot in range(args.batch)
        ]
        print("BATCH_PREFILL_COMPARISONS", comparisons, flush=True)
        if args.diagnose:

            def read(value):
                return ttnn.to_torch(ttnn.get_device_tensors(value)[0])

            hidden = read(generator.prefill[64]["hidden"])
            sliced = ttnn.slice(
                generator.prefill[64]["hidden"], generator.slice_start, generator.slice_end, slice_dim=2, num_devices=2
            )
            host_slice = read(sliced)
            selected = ttnn.embedding(generator.last_indices, sliced, layout=ttnn.TILE_LAYOUT)
            host_selected = read(selected)
            selected = ttnn.reshape(selected, (1, 1, args.batch, 6656))
            terminal = generator.model.head(selected, decode=True)
            host_terminal = generator._read_logits(terminal)
            details = dict(
                comparisons=comparisons,
                slice_equals_expected=torch.equal(host_slice, hidden[:, :, 32:64]),
                selected_equals_expected=torch.equal(
                    host_selected.reshape(args.batch, 6656), hidden[0, 0, 32:33].expand(args.batch, -1)
                ),
                selected_max_abs=(host_selected.reshape(args.batch, 6656).float() - hidden[0, 0, 32:33].float())
                .abs()
                .max()
                .item(),
                terminal_argmax=host_terminal.argmax(-1).flatten().tolist(),
                terminal_range=[host_terminal.min().item(), host_terminal.max().item()],
                hidden_range=[hidden.min().item(), hidden.max().item()],
            )
            torch.save(
                dict(prefill=first, hidden=hidden, sliced=host_slice, selected=host_selected, terminal=host_terminal),
                args.output.with_suffix(".pt"),
            )
            args.output.write_text(json.dumps(details, indent=2) + "\n")
            print("BATCH_BOUNDARY_DETAILS", details, flush=True)
            return
        for slot in range(2, args.batch):
            assert torch.equal(first[slot], first[slot % 2]), ("prefill slot", slot)
        values = first.argmax(-1).flatten().tolist()
        second = generator.decode_forward(values, lengths, page_table=table, kv_cache=generator.kv_cache, sample=False)
        for slot in range(2, args.batch):
            assert torch.equal(second[slot], second[slot % 2]), ("decode slot", slot)
        # An inactive row must retain its position and KV. Reactivation gives
        # the same result as a second active slot at the same token/position.
        same = [33] * args.batch
        generator.reset()
        baseline = generator.prefill_forward(
            ids, page_table=table, kv_cache=generator.kv_cache, prompt_lens=same, sample=False
        )
        token = baseline.argmax(-1).flatten().tolist()
        inactive = [33] * args.batch
        inactive[-1] = -1
        active_out = generator.decode_forward(
            token, inactive, page_table=table, kv_cache=generator.kv_cache, sample=False
        )
        reactivation = [-1] * args.batch
        reactivation[-1] = 33
        resumed = generator.decode_forward(
            token, reactivation, page_table=table, kv_cache=generator.kv_cache, sample=False
        )
        assert torch.equal(active_out[0], resumed[-1]), "Inactive slot cache changed"
        # Partial append at a nonaligned start uses the same decoder trace.
        append_ids = torch.arange(32, 161)[None].expand(args.batch, -1).clone()
        generator.reset()
        full = generator.prefill_forward(
            append_ids, page_table=table, kv_cache=generator.kv_cache, prompt_lens=[129] * args.batch, sample=False
        )
        generator.reset()
        generator.prefill_forward(
            append_ids[:, :31],
            page_table=table,
            kv_cache=generator.kv_cache,
            prompt_lens=[31] * args.batch,
            sample=False,
        )
        appended = generator.prefill_forward(
            append_ids[:, 31:],
            page_table=table,
            kv_cache=generator.kv_cache,
            prompt_lens=[98] * args.batch,
            start_pos=[31] * args.batch,
            sample=False,
        )
        agreement = torch.nn.functional.cosine_similarity(full.float(), appended.float(), dim=-1)
        append_control = dict(
            command=[sys.executable, *sys.argv],
            precision_config=generator.model.precision_config,
            precision_config_environment_override=os.environ.get("MUSE_PRECISION_CONFIG"),
            implementation_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")
            },
            layers=2 if args.reduced else 52,
            batch=args.batch,
            contiguous_length=129,
            append_lengths=[31, 98],
            cosine_min=agreement.min().item(),
            finite_logits=bool(torch.isfinite(full).all() and torch.isfinite(appended).all()),
            contiguous_tokens=full.argmax(-1).flatten().tolist(),
            appended_tokens=appended.argmax(-1).flatten().tolist(),
            argmax_equal=torch.equal(full.argmax(-1), appended.argmax(-1)),
            captures=generator.counters["captures"],
        )
        append_logits = args.output.with_suffix(".append.pt")
        torch.save(dict(contiguous=full, appended=appended), append_logits)
        append_control["logits_sha256"] = hashlib.sha256(append_logits.read_bytes()).hexdigest()
        append_control["status"] = (
            "pass" if agreement.min() > 0.99 and append_control["argmax_equal"] else "capability_gate_fail"
        )
        args.output.with_suffix(".append.json").write_text(json.dumps(append_control, indent=2) + "\n")
        print("BATCH_APPEND_CONTROL", append_control, flush=True)
        assert agreement.min() > 0.99, agreement
        assert torch.equal(full.argmax(-1), appended.argmax(-1))
        generator._execute("sample")
        feedback = generator._read_tokens()
        generator._execute("decode")
        generator._execute("sample")
        next_tokens = generator._read_tokens()
        assert len(next_tokens) == args.batch
        assert len(set(next_tokens)) == 1, next_tokens
        generator.reset()
        permuted = generator.prefill_forward(
            ids[:2],
            page_table=table,
            kv_cache=generator.kv_cache,
            prompt_lens=[31, 33],
            slots=[args.batch - 1, 0],
            sample=False,
        )
        assert torch.equal(permuted[-1], first[0]) and torch.equal(permuted[0], first[1])
        assert not torch.count_nonzero(permuted[1:-1]), "Unused slots retained another request's logits"
        assert generator._host_positions == [33] + [-1] * (args.batch - 2) + [31]
        generator._execute("sample")
        queued_first = generator._read_tokens()
        before_steps = dict(generator.counters)
        generator.decode_steps(3)
        step_counts = {k: v - before_steps.get(k, 0) for k, v in generator.counters.items()}
        assert all(
            step_counts.get(k, 0) == 0
            for k in (
                "token_refreshes",
                "position_refreshes",
                "rope_refreshes",
                "page_table_refreshes",
                "token_readbacks",
                "async_token_readbacks",
                "full_logits_readbacks",
                "synchronizations",
                "captures",
            )
        )
        queued_tokens = generator._read_tokens()
        queued_positions = ttnn.to_torch(ttnn.get_device_tensors(generator.positions)[0]).flatten().tolist()
        assert queued_positions == [36] + [-1] * (args.batch - 2) + [34]
        assert generator._host_positions == queued_positions
        # Admit a prompt into another slot after queued replay. Existing active
        # positions must survive the prefill's scheduler-state upload.
        admission_positions = None
        if args.batch > 2:
            generator.prefill_forward(
                ids[:1],
                page_table=table,
                kv_cache=generator.kv_cache,
                prompt_lens=[31],
                slots=[1],
                sample=False,
            )
            admission_positions = ttnn.to_torch(ttnn.get_device_tensors(generator.positions)[0]).flatten().tolist()
            expected = queued_positions.copy()
            expected[1] = 31
            assert admission_positions == expected
        before_rejection = dict(generator.counters)
        try:
            generator.decode_steps(generator.model.config.max_position_embeddings)
        except ValueError:
            pass
        else:
            raise AssertionError("Over-context replay was admitted")
        assert dict(generator.counters) == before_rejection
        generator.reset()
        control_first = (
            generator.prefill_forward(
                ids[:2], page_table=table, kv_cache=generator.kv_cache, prompt_lens=[31, 33], slots=[args.batch - 1, 0]
            )
            .flatten()
            .tolist()
        )
        control_steps = []
        for _ in range(3):
            generator._execute("decode")
            generator._execute("sample")
            control_tokens = generator._read_tokens()
            control_steps.append(control_tokens)
        print(
            "QUEUED_CONTROL",
            dict(first=[queued_first, control_first], queued=queued_tokens, control=control_steps),
            flush=True,
        )
        # Position -1 suppresses KV writes/attention. Its output logits/tokens
        # are not request output and may contain untouched attention scratch.
        # Compare exactly the two admitted requests; inactive cache preservation
        # and reactivation are checked above on the full stack.
        active_slots = [0, args.batch - 1]
        assert [queued_tokens[i] for i in active_slots] == [control_tokens[i] for i in active_slots], (
            queued_tokens,
            control_tokens,
        )
        cache_reset_equivalence = check_cache_reset_equivalence(generator)
        live_admission = check_live_admission(generator)
        result = dict(
            status="pass",
            command=[sys.executable, *sys.argv],
            precision_config=generator.model.precision_config,
            precision_config_environment_override=os.environ.get("MUSE_PRECISION_CONFIG"),
            live_admission=live_admission,
            cache_reset_equivalence=cache_reset_equivalence,
            terminal_policy=generator.model.terminal_policy,
            implementation_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")
            },
            batch=args.batch,
            layers=2 if args.reduced else 52,
            mixed_prompt_lengths=lengths,
            identical_logits_across_slots=True,
            inactive_cache_preserved=True,
            reordered_slots_preserved=True,
            append_cosine_min=agreement.min().item(),
            append_argmax_equal=append_control["argmax_equal"],
            feedback=feedback,
            next_tokens=next_tokens,
            no_readback_fixed_slots=dict(
                steps=3,
                counters=step_counts,
                positions=queued_positions,
                active_slots=active_slots,
                active_tokens_match_readback_control=True,
                inactive_tokens_are_not_outputs=True,
                positions_after_admitting_slot1=admission_positions,
                over_context_rejected_before_dispatch=True,
                final_tokens=queued_tokens,
            ),
            counters=dict(generator.counters),
            captures=generator.TRACE_COUNT,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("BATCH_CONTRACT_PASS", result, flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if generator:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--diagnose-eager", action="store_true")
    parser.add_argument("--diagnose-ops", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
