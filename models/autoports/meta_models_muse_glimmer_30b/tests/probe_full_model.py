# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fast real-shape wrapper probe; never substitutes for the all-layer gate."""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.model import MuseGlimmerModel
from models.common.modules.sampling.sampling_1d import Sampling1D


@contextmanager
def device_only_audit():
    old = {name: getattr(ttnn, name) for name in ("from_torch", "to_torch", "as_tensor", "copy_host_to_device_tensor")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Host boundary inside a captured device graph")

    def profile(frame, event, arg):
        module = frame.f_globals.get("__name__", "")
        if event == "call" and (module == "torch" or module.startswith("torch.")):
            raise AssertionError(f"Torch computation inside device graph: {module}.{frame.f_code.co_name}")

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        for name in old:
            setattr(ttnn, name, forbidden)
        yield
    finally:
        sys.setprofile(previous)
        for name, value in old.items():
            setattr(ttnn, name, value)


def main(args):
    torch.set_num_threads(8)
    started = time.perf_counter()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        model = MuseGlimmerModel(mesh, layer_indices=[0, 3])
        if args.head_block is not None:
            model.lm_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=args.head_block,
                per_core_M=1,
                per_core_N=math.ceil(
                    model.padded_vocab_size // 4 / (32 * model.lm_input_mem.shard_spec.grid.num_cores())
                ),
                num_workers_per_dram_bank=args.head_readers,
            )
            model.lm_head.config.program_configs = [model.lm_program]
        if args.generator:
            from models.autoports.meta_models_muse_glimmer_30b.tt.generator import MuseGlimmerGenerator

            if args.audit:

                def wrap(method):
                    def audited(*positional, **keywords):
                        with device_only_audit():
                            return method(*positional, **keywords)

                    return audited

                for name in (
                    "_prefill_graph",
                    "_head_graph",
                    "_decode_graph",
                    "_sample_graph",
                    "_reset_cache_graph",
                    "_save_admission_graph",
                    "_restore_admission_graph",
                ):
                    setattr(MuseGlimmerGenerator, name, wrap(getattr(MuseGlimmerGenerator, name)))
            external_cache = (
                model.allocate_cache(batch_size=args.batch, num_pages=args.pages) if args.external_cache else None
            )
            generator = MuseGlimmerGenerator(
                model, batch_size=args.batch, num_pages=args.pages, kv_cache=external_cache
            )
            if external_cache is not None:
                assert generator.kv_cache is external_cache
            print("GENERATOR_PREPARED", generator.prepare_seconds, flush=True)
            outputs = []
            for length in (31, 33, 127, 129):
                outputs.append(generator.generate(list(range(32, 32 + length)), 4))
                print("GENERATED", length, outputs[-1], flush=True)
            reference = torch.load(Path(__file__).parents[1] / "doc/full_model/probe_hf.pt", weights_only=False)
            generator.reset()
            table = generator._standalone_table()
            prefill = generator.prefill_forward(
                torch.arange(32, 64)[None],
                page_table=table,
                kv_cache=generator.kv_cache,
                prompt_lens=[32],
                sample=False,
            )
            print("CONTRACT_PREFILL", prefill.shape, prefill.flatten()[:12], prefill.argmax(-1), flush=True)
            if args.diagnose_prefill:

                def inspect(name, value):
                    host = ttnn.to_torch(ttnn.get_device_tensors(value)[0])
                    print(
                        "BOUNDARY",
                        name,
                        list(host.shape),
                        host.float().min().item(),
                        host.float().max().item(),
                        host.flatten()[:12],
                        flush=True,
                    )
                    return host

                state = generator.prefill[32]
                inspect("hidden_trace", state["hidden"])
                inspect("final_norm", model.final_norm)
                selected = ttnn.embedding(generator.last_indices, state["hidden"], layout=ttnn.TILE_LAYOUT)
                inspect("selected", selected)
                selected = ttnn.to_memory_config(ttnn.reshape(selected, (1, 1, args.batch, 6656)), model.lm_input_mem)
                norm = ttnn.rms_norm(
                    selected,
                    epsilon=model.config.rms_norm_eps,
                    weight=model.final_norm,
                    program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=[8, 8], subblock_w=1, block_h=1, block_w=4, inplace=False
                    ),
                    compute_kernel_config=model.compute_config,
                    memory_config=model.lm_input_mem,
                )
                inspect("normalized", norm)
                linear = ttnn.linear(
                    norm,
                    model.lm_decode,
                    program_config=model.lm_program,
                    compute_kernel_config=model.compute_config,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )
                inspect("raw_linear", linear)
                inspect("head_direct", model.head(state["hidden"], decode=True))
                del selected, norm, linear
                inspect("row_mask", generator.row_mask)
                inspect("last_indices", generator.last_indices)
                inspect("pending_logits", generator.prefill_logits_buffer)
                generator._head_graph(32)
                inspect("pending_logits_eager_head", generator.prefill_logits_buffer)
                inspect("head_all", generator._head_graph(32, all_rows=True))
                generator._prefill_graph(32)
                inspect("hidden_eager", state["hidden"])
                generator._head_graph(32)
                inspect("pending_logits_eager", generator.prefill_logits_buffer)
                generator.close()
                return
            decoded = generator.decode_forward(
                torch.full((args.batch,), 42),
                [32] + [-1] * (args.batch - 1),
                page_table=table,
                kv_cache=generator.kv_cache,
                sample=False,
            )
            accuracy = {}
            for name, actual in (("prefill", prefill[0]), ("decode", decoded[0])):
                expected = reference[name].flatten().float()
                actual = actual.flatten().float()
                pcc = torch.corrcoef(torch.stack((actual, expected)))[0, 1].item()
                accuracy[name] = dict(pcc=pcc, token=int(actual.argmax()), hf_token=int(expected.argmax()))
                # The full-model gate is top-k accuracy. Logit PCC is a
                # diagnostic, calibrated against the selected eager decoder.
                eager_path = args.eager_control or (
                    Path(__file__).parents[1] / "doc/full_model/probe_eager_selected_control.pt"
                )
                if args.eager_control:
                    assert eager_path.exists(), "Requested same-policy eager control is missing"
                    eager_metadata = json.loads(eager_path.with_suffix(".json").read_text())
                    assert eager_metadata["precision_config"] == model.precision_config
                    assert eager_metadata["eager_control_sha256"] == hashlib.sha256(eager_path.read_bytes()).hexdigest()
                if eager_path.exists():
                    eager = torch.load(eager_path, weights_only=False)[name].flatten().float()[: model.vocab_size]
                    eager_pcc = torch.corrcoef(torch.stack((eager, expected)))[0, 1].item()
                    trace_eager_pcc = torch.corrcoef(torch.stack((eager, actual)))[0, 1].item()
                    accuracy[name].update(eager_hf_pcc=eager_pcc, trace_eager_pcc=trace_eager_pcc)
                    assert pcc >= eager_pcc - 0.001 and trace_eager_pcc > 0.999, accuracy
                assert int(actual.argmax()) in expected.topk(5).indices.tolist(), accuracy
            altered = generator.decode_forward(
                torch.full((args.batch,), 43),
                [32] + [-1] * (args.batch - 1),
                page_table=table,
                kv_cache=generator.kv_cache,
                sample=False,
            )
            assert not torch.equal(altered, decoded), "Changed token was ignored"
            restored = generator.decode_forward(
                torch.full((args.batch,), 42),
                [32] + [-1] * (args.batch - 1),
                page_table=table,
                kv_cache=generator.kv_cache,
                sample=False,
            )
            assert torch.equal(restored, decoded), "Refreshed token/position failed deterministic replay"

            def read(tensor):
                return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).flatten().tolist()

            generator._execute("sample")
            previous = read(generator.tokens)
            before = dict(generator.counters)
            feedback = []
            for index in range(3):
                consumed = read(generator.tokens)
                assert consumed == previous
                generator._execute("decode")
                assert read(generator.positions) == [34 + index] + [-1] * (args.batch - 1)
                generator._execute("sample")
                previous = read(generator.tokens)
                feedback.append(dict(consumed=consumed, produced=previous, positions=read(generator.positions)))
            for counter in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
                assert generator.counters[counter] == before.get(counter, 0)
            page_copies = generator.counters["page_table_refreshes"]
            generator._refresh_table(table)
            assert generator.counters["page_table_refreshes"] == page_copies
            changed = table.flip(-1)
            generator._refresh_table(changed)
            assert read(generator.page_table) == changed.flatten().tolist()
            assert generator.counters["page_table_refreshes"] == page_copies + 1
            moved_prefill = generator.prefill_forward(
                torch.arange(32, 64)[None],
                page_table=changed,
                kv_cache=generator.kv_cache,
                prompt_lens=[32],
                sample=False,
            )
            moved_decode = generator.decode_forward(
                torch.full((args.batch,), 42),
                [32] + [-1] * (args.batch - 1),
                page_table=changed,
                kv_cache=generator.kv_cache,
                sample=False,
            )
            assert torch.equal(moved_prefill, prefill) and torch.equal(moved_decode, decoded)
            generator._refresh_table(table)
            # All signatures were captured before requests and survive resets.
            trace_ids = {str(k): str(v) for k, v in generator.traces.items()}
            # The last two cover a padded/exact 64-row tail at nonzero start.
            for length in (257, 513, 1025, 2049, 4097, 4129, 4160):
                generator.generate([42] * length, 3, stop_on_eos=False)
            sample_output = generator.generate([42] * 33, 4, top_k=8, top_p=0.9, stop_on_eos=False)
            seeded = [
                generator.generate([42] * 33, 16, top_k=8, top_p=0.9, seeds=seed, stop_on_eos=False)
                for seed in (1234, 4321, 1234)
            ]
            assert seeded[0] == seeded[2] and seeded[0] != seeded[1]
            greedy = generator.generate(list(range(32, 65)), 4, stop_on_eos=False)
            argmax = generator.generate(list(range(32, 65)), 4, sampler_mode="argmax", stop_on_eos=False)
            assert greedy == argmax == outputs[1], (greedy, argmax, outputs[1])
            compatibility = generator.generate(list(range(32, 65)), 4, host_sampling=True, stop_on_eos=False)
            assert compatibility == greedy, (compatibility, greedy)
            assert trace_ids == {str(k): str(v) for k, v in generator.traces.items()}
            assert generator.counters["captures"] == generator.TRACE_COUNT
            result = dict(
                command=[sys.executable, *sys.argv],
                precision_config=model.precision_config,
                precision_config_environment_override=os.environ.get("MUSE_PRECISION_CONFIG"),
                implementation_sha256={
                    str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__).parents[1] / "tt").glob("*.py")
                },
                terminal_policy=generator.model.terminal_policy,
                outputs=outputs,
                accuracy=accuracy,
                feedback=feedback,
                sampled=sample_output,
                seeded_outputs=seeded,
                runtime_fallback_audit=args.audit,
                eager_control_provenance=dict(
                    path=str(eager_path),
                    sha256=hashlib.sha256(eager_path.read_bytes()).hexdigest() if eager_path.exists() else None,
                    policy=(
                        model.precision_config
                        if args.eager_control
                        else "completed stage6 decoder and BF16/HiFi4 LM head"
                    ),
                    note=(
                        "fresh same-policy eager control; original PCC and HF top5 gates retained"
                        if args.eager_control
                        else "prior-policy eager control"
                    ),
                ),
                external_cache=args.external_cache,
                host_sampling_compatibility=compatibility,
                greedy=greedy,
                argmax=argmax,
                counters=dict(generator.counters),
                traces=trace_ids,
                elapsed_seconds=time.perf_counter() - started,
            )
            generator.close()
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            return
        cache = model.allocate_cache(batch_size=args.batch, num_pages=args.pages)
        host_table = torch.arange(args.pages, dtype=torch.int32).reshape(args.batch, -1)
        table = model.upload(host_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ids = torch.arange(32, 32 + args.length, dtype=torch.int32).reshape(1, -1)
        print("PREFILL_START", flush=True)
        from types import SimpleNamespace

        from models.autoports.meta_models_muse_glimmer_30b.tt.generator import MuseGlimmerGenerator

        with MuseGlimmerGenerator._prefill_policy(SimpleNamespace(model=model, mesh=mesh), args.length):
            logits = model.prefill_forward(
                model.upload(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                kv_cache=cache,
                page_table=table,
                all_logits=False,
            )
        host_logits = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        prefill_control = host_logits.clone()
        print("PREFILL_OK", list(logits.shape), host_logits.flatten()[:4], flush=True)
        input_ids = model.upload(
            torch.full((1, 1, 1, args.batch), 42, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        positions = model.upload(
            torch.full((args.batch,), args.length, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        rope_indices = model.upload(
            torch.full((1, args.batch), args.length, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        print("DECODE_START", flush=True)
        logits = model.decode_forward(
            input_ids, kv_cache=cache, page_table=table, current_pos=positions, rope_indices=rope_indices
        )
        host_logits = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        torch.save(
            dict(prefill=prefill_control[..., : model.vocab_size], decode=host_logits[..., : model.vocab_size]),
            args.output.with_suffix(".pt"),
        )
        print("DECODE_OK", list(logits.shape), host_logits.flatten()[:4], flush=True)
        sampler = Sampling1D(
            model.padded_vocab_size,
            mesh_device=mesh,
            valid_vocab_size=model.vocab_size,
            tt_ccl=model.terminal_ccl,
            max_batch_size=args.batch,
            max_top_k=32,
            pad_to_power_of_2=True,
            num_gather_links=2,
            allow_force_argmax=True,
            ag_topology=ttnn.Topology.Ring,
        )
        k = model.upload(torch.ones(args.batch, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        p = model.upload(torch.zeros(args.batch), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        temp = model.upload(torch.ones(args.batch), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        print("SAMPLING_START", flush=True)
        token, _ = sampler.decode_forward(logits, k=k, p=p, temp=temp, tt_out_tok=input_ids)
        tokens = ttnn.to_torch(ttnn.get_device_tensors(token)[0]).flatten().tolist()
        expected = host_logits[..., : model.vocab_size].argmax(-1).flatten().tolist()
        print("SAMPLING_OK", tokens, expected, flush=True)
        assert tokens[: args.batch] == expected[: args.batch]
        output = dict(
            command=[sys.executable, *sys.argv],
            precision_config_environment_override=os.environ.get("MUSE_PRECISION_CONFIG"),
            implementation_sha256={
                str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__).parents[1] / "tt").glob("*.py")
            },
            precision_config=model.precision_config,
            terminal_policy=model.terminal_policy,
            eager_control_sha256=hashlib.sha256(args.output.with_suffix(".pt").read_bytes()).hexdigest(),
            layers=[0, 3],
            length=args.length,
            batch=args.batch,
            tokens=tokens,
            expected_tokens=expected,
            elapsed_seconds=time.perf_counter() - started,
        )
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--pages", type=int, default=4096)
    parser.add_argument("--generator", action="store_true")
    parser.add_argument("--diagnose-prefill", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--external-cache", action="store_true")
    parser.add_argument("--eager-control", type=Path, help="Same-policy eager logits and sibling JSON provenance")
    parser.add_argument("--head-block", type=int)
    parser.add_argument("--head-readers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
