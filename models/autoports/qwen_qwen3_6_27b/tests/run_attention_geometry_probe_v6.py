# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V6: packed projection with exact-K-divisor C40/C80 storage and intermediate readers.

Select the decoder with evidence.decoder_class/decoder_kwargs environment variables.
Host capture/reference/readback occurs outside all timed traces. No model fallback.
"""
import argparse
import csv
import gc
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from unittest.mock import patch

KEYS = ('packed_gdn',)


def geometry_plan(k, n, banks):
    """Eight cases that satisfy both factory and compute_output_specs rules."""
    assert (k, n) == (5120, 16512)
    configurations = [(32, 4), (8, 5), (8, 6), (16, 6),
                      (40, 5), (40, 10), (80, 5), (80, 10)]
    rows = []
    for cores, readers in configurations:
        assert k % (32 * cores) == 0
        shard_k_tiles = k // (32 * cores)
        # compute_output_specs rejects uneven sharding even when the
        # multicore factory accepts excess capacity: both constraints matter.
        assert (k // 32) % shard_k_tiles == 0
        active_input_cores = (k // 32) // shard_k_tiles
        assert active_input_cores == cores
        alignment = math.lcm(banks * readers, cores)
        physical_n = math.ceil(n / (32 * alignment)) * 32 * alignment
        bank_tiles = physical_n // (32 * banks)
        column_workers = min(readers, bank_tiles)
        k_groups = min(readers // bank_tiles, k // 32) if readers > bank_tiles else 1
        max_columns = math.ceil(bank_tiles / column_workers)
        passes = math.ceil(max_columns / 8)
        assert k_groups == 1
        assert active_input_cores <= cores
        assert ((k // 32) - 1) // shard_k_tiles == active_input_cores - 1
        rows.append(dict(cores=cores, readers=readers, block=1,
                         physical_n=physical_n, shard_k_tiles=shard_k_tiles,
                         input_shard_width=32 * shard_k_tiles,
                         expected_active_input_cores=active_input_cores,
                         unused_input_grid_cores=cores-active_input_cores,
                         weight_shard_n=physical_n // banks,
                         per_core_N=physical_n // 32 // cores, pipeline=True,
                         expected_compute_workers=banks * column_workers * k_groups,
                         expected_k_groups=k_groups,
                         expected_max_columns_per_worker=max_columns,
                         expected_dst_column_passes=passes,
                         weight_padding_columns=physical_n-n))
    assert len(rows) == 8
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--key', action='append', choices=KEYS, default=[KEYS[0]])
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--batch', type=int, choices=(1, 3, 32), default=32)
    ap.add_argument('--prefix-length', type=int, default=33)
    ap.add_argument('--capture-step', type=int, default=3, help='Zero-based decode step after real prefill')
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--trace-bytes', type=int, default=16 * 1024 * 1024)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--plan-only', action='store_true', help='No TTNN import or device access; assumes eight DRAM banks')
    args = ap.parse_args()
    assert args.prefix_length >= 1 and args.capture_step >= 0 and args.iterations >= 3
    args.key = list(dict.fromkeys(args.key))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        shapes = {'packed_gdn': (5120, 16512)}
        args.output.write_text(json.dumps({key: geometry_plan(*shapes[key], 8) for key in args.key}, indent=2) + '\n')
        return

    import torch
    import ttnn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class, decoder_kwargs, provenance
    from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
    from models.autoports.qwen_qwen3_6_27b.tests.run_projection_precision_probe import (
        compare, cpu_projection, fingerprint, original_weight,
    )

    torch.set_num_threads(8)
    torch.manual_seed(17)
    prov = provenance()
    assert prov['decoder_implementation'] == 'optimized'
    config = Qwen3_5TextConfig(**json.loads((Path(__file__).parents[1] / 'doc/hf_config.json').read_text())['text_config'])
    weights = load_layer(args.layer)
    report = dict(provenance=prov, arguments=vars(args) | {'output': str(args.output)},
                  scope='Full corrected projection: input reshard + two FP32 linear calls + BF16 add + DRAM conversion + logical slice',
                  precision='actual captured BFP8 primary/residual, HiFi4, FP32 DST/output, packer_l1_acc=False, math_approx=False',
                  capture='actual compact decode ttnn.linear operands identified by layer.weights object identity; immediate host snapshots retain no activation/partial L1 allocation',
                  references='FP64 products of stored prequantized values and original BF16 checkpoint weights',
                  model_acceptance='Diagnostic projection comparison; decoder PCC gates unchanged', rows=[], operands={})

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        columns = ('key', 'control', 'status', 'cores', 'readers', 'block', 'physical_n',
                   'trace_median_ms', 'trace_min_ms', 'replay_bit_equal', 'weight_elements_exact',
                   'stored_pair_pcc', 'checkpoint_pcc', 'error_type')
        with args.output.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in report['rows']:
                flat = {name: row.get(name) for name in columns}
                for source, dest in (('final_vs_stored_pair_fp64', 'stored_pair_pcc'),
                                     ('final_vs_checkpoint_fp64', 'checkpoint_pcc')):
                    metrics = row.get(source, {})
                    flat[dest] = metrics.get('all_valid_channels', metrics.get('all', {})).get('pcc')
                writer.writerow(flat)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=args.trace_bytes)
    try:
        layer = decoder_class().from_state_dict(weights, hf_config=config, layer_idx=args.layer,
                                                mesh_device=mesh, **decoder_kwargs())
        assert layer.kind == 'linear_attention', 'This probe targets corrected GDN attention projections only'
        assert layer.policy.get('attention_fidelity') == 'HiFi4'
        assert layer.policy.get('attention_packer_l1_acc', layer.policy.get('packer_l1_acc', True)) is False
        assert layer.policy.get('correction_l1') and layer.policy.get('dram_sharded')
        composer = ttnn.ConcatMeshToTensor(mesh, dim=0)

        def read(tensor):
            tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG) if tensor.is_sharded() else tensor
            return ttnn.to_torch(tensor, mesh_composer=composer).detach().cpu().clone()

        selected = [k for key in args.key for k in (key, layer.weight_corrections[key])]
        for key in args.key:
            correction = layer.weight_corrections[key]
            assert correction == key + '_residual'
            for part in (key, correction):
                assert layer.weights[part].dtype == ttnn.bfloat8_b
                pc = layer.projection_info[part]['program']
                assert pc.num_workers_per_dram_bank == 4 and pc.in0_block_w == 1 and pc.use_multicore_pipeline
                assert layer.projection_info[part]['input_mem'].shard_spec.grid.num_cores() == 32
        state = layer.create_recurrent_state(args.batch)
        prefix = (torch.randn(args.batch, args.prefix_length, 5120) * .1).bfloat16()
        layer.prefill_forward(layer._upload(prefix), state=state)
        captured = {}
        native_linear = ttnn.linear

        def record_linear(*operands, **kw):
            result = native_linear(*operands, **kw)
            weight = operands[1] if len(operands) > 1 else kw.get('weight')
            for key in selected:
                if weight is layer.weights[key]:
                    assert key not in captured, ('duplicate captured linear', key)
                    assert result.dtype == ttnn.float32 and kw.get('dtype') == ttnn.float32
                    # Retaining these L1 tensors through _finish changes the model's
                    # allocation lifetime and clashes with the MLP down CB region.
                    # Snapshot now, outside all timed windows; retain only DRAM
                    # weight identity plus non-tensor program/config metadata.
                    metadata = {name: kw[name] for name in
                                ('program_config', 'compute_kernel_config', 'memory_config', 'dtype')}
                    captured[key] = (read(operands[0]), weight, metadata, read(result))
            return result

        for step in range(args.capture_step + 1):
            host = (torch.randn(1, args.batch, 5120) * .1).bfloat16()
            token = layer._upload(host)
            if step == args.capture_step:
                with patch.object(ttnn, 'linear', record_linear):
                    result = layer.decode_forward(token, state=state)
            else:
                result = layer.decode_forward(token, state=state)
            # Neither a prior decode result nor its input belongs in a projection
            # geometry's live allocation set. State remains alive through capture.
            del result, token
        gc.collect()
        assert set(captured) == set(selected), ('missing actual linear operands', set(selected) - set(captured))
        report['effective_policy'] = layer.policy
        report['hypothesis'] = ('Use exact K divisors C40/shard4 and C80/shard2 plus intermediate R5/R6 controls; fixed K1, logical K5120 and no K split. '
                                'Factory-predicted DST pass reduction is a hypothesis, not performance evidence.')
        grid = mesh.compute_with_storage_grid_size()
        dg = mesh.dram_grid_size()
        banks = dg.x * dg.y
        report['device_worker_grid'] = [grid.x, grid.y]
        report['device_dram_banks'] = banks
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))})
        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
        saved_native = {}
        saved_inputs = {}

        def measure(run):
            outputs = run()
            actual = [read(t) for t in outputs]
            del outputs
            for _ in range(3):
                warm = run()
                ttnn.synchronize_device(mesh)
                del warm
            trace = None
            try:
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                outputs = run()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                for _ in range(3):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                elapsed = []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    elapsed.append((time.perf_counter() - start) * 1000)
                replay = [read(t) for t in outputs]
                return actual, dict(trace_median_ms=statistics.median(elapsed), trace_min_ms=min(elapsed),
                                    trace_samples_ms=elapsed, replay_bit_equal=all(torch.equal(a, b) for a, b in zip(actual, replay)))
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)

        for key in args.key:
            correction = layer.weight_corrections[key]
            a0, w0, kw0, y0 = captured[key]
            a1, w1, kw1, y1 = captured[correction]
            host_x = a0.reshape(1, args.batch, -1).bfloat16()
            assert torch.equal(host_x, a1.reshape_as(host_x))
            saved_inputs[key] = host_x
            capture_path = args.output.with_suffix('.inputs.pt')
            torch.save({'inputs': saved_inputs, 'prefix_length': args.prefix_length,
                        'capture_step': args.capture_step, 'seed': 17, 'batch': args.batch}, capture_path)
            report['captured_input_artifact'] = str(capture_path)
            logical_n = layer.projection_info[key]['logical_n']
            k = host_x.shape[-1]
            original = original_weight(weights, key)
            assert tuple(original.shape) == (k, logical_n)
            stored = [read(w).float()[:, :logical_n].contiguous() for w in (w0, w1)]
            references = [cpu_projection(host_x, w) for w in stored]
            pair_reference = references[0] + references[1]
            checkpoint_reference = cpu_projection(host_x, original)
            # Exact BF16 snapshot re-upload is probe setup, not measured runtime.
            # Measured projection still performs the original input reshard.
            tx = layer._upload(host_x)
            base_weights = [ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG) for w in (w0, w1)]
            info = dict(input_shape=list(host_x.shape), input_sha256=fingerprint(host_x),
                        weight_sha256=[fingerprint(w) for w in stored],
                        checkpoint_weight_sha256=fingerprint(original),
                        captured_config=str(kw0['compute_kernel_config']),
                        captured_program=str(kw0['program_config']), logical_n=logical_n,
                        native_partial=[compare(references[i], y.reshape(args.batch, -1)[:, :logical_n], key)
                                        for i, y in enumerate((y0, y1))])
            report['operands'][key] = info
            native_mem = layer.projection_info[key]['input_mem']

            def projection(input_mem, device_weights, programs):
                activation = ttnn.to_memory_config(tx, input_mem)
                partial = [ttnn.linear(activation, w, program_config=pc,
                                      compute_kernel_config=kw0['compute_kernel_config'],
                                      memory_config=output_mem, dtype=ttnn.float32)
                           for w, pc in zip(device_weights, programs)]
                final = ttnn.add(*partial, dtype=ttnn.bfloat16)
                final = ttnn.to_memory_config(final, ttnn.DRAM_MEMORY_CONFIG)[:, :, :logical_n]
                return *partial, final

            def record(row, run):
                try:
                    actual, timing = measure(run)
                    values = [a.reshape(args.batch, -1)[:, :logical_n] for a in actual]
                    row.update(timing)
                    row['partial_vs_stored_fp64'] = [compare(ref, val, key) for ref, val in zip(references, values[:2])]
                    row['final_vs_stored_pair_fp64'] = compare(pair_reference, values[2], key)
                    row['final_vs_checkpoint_fp64'] = compare(checkpoint_reference, values[2], key)
                    if key not in saved_native:
                        saved_native[key] = values
                    row['vs_native_control'] = [compare(ref, val, key) for ref, val in zip(saved_native[key], values)]
                    row['native_bit_equal'] = [torch.equal(ref, val) for ref, val in zip(saved_native[key], values)]
                    row['status'] = 'measured'
                except RuntimeError as exc:
                    row.update(status='runtime_error', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
                report['rows'].append(row)
                save()

            record(dict(key=key, control='native_c32_r4_b1', cores=32, readers=4, block=1,
                        physical_n=w0.shape[-1], pipeline=True, weight_elements_exact=True),
                   lambda: projection(native_mem, (w0, w1), (kw0['program_config'], kw1['program_config'])))
            assert key in saved_native, 'Native baseline failed; no candidate comparison is valid'
            cached_geometry = None
            sw = None
            for geometry in geometry_plan(k, logical_n, banks):
                row = dict(key=key, control=None, **geometry)
                identity = (geometry['cores'], geometry['readers'], geometry['physical_n'])
                try:
                    if identity != cached_geometry:
                        sw = None
                        gc.collect()
                        cores = ttnn.num_cores_to_corerangeset(geometry['cores'], grid, row_wise=True)
                        input_mem = ttnn.create_sharded_memory_config((32, geometry['input_shard_width']), cores,
                            ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True)
                        # Exact K-divisor storage: no empty input grid cores,
                        # logical K padding, or changed activation values.
                        checked_activation = ttnn.to_memory_config(tx, input_mem)
                        assert tuple(checked_activation.shape) == tuple(tx.shape)
                        assert checked_activation.padded_shape[-1] == k
                        assert torch.equal(read(checked_activation).reshape_as(host_x), host_x)
                        active_input_cores = len(ttnn.get_optimal_worker_cores_for_sharded_tensor(checked_activation))
                        assert active_input_cores == geometry['expected_active_input_cores']
                        del checked_activation
                        weight_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                            ttnn.ShardSpec(dram_grid, (k, geometry['weight_shard_n']), ttnn.ShardOrientation.ROW_MAJOR))
                        transformed = []
                        transformations = []
                        for index, source in enumerate(base_weights):
                            n = geometry['physical_n']
                            method = 'tile_aligned_device_slice_pad'
                            try:
                                prepared = source[:, :n] if source.shape[-1] > n else source
                                if prepared.shape[-1] < n:
                                    prepared = ttnn.pad(prepared, ((0, 0), (0, n-prepared.shape[-1])), value=0.0)
                                prepared = ttnn.to_memory_config(prepared, weight_mem)
                            except RuntimeError as exc:
                                # Setup-only fallback: retain exact stored values, not checkpoint requantization.
                                method = 'host_stored_values_reupload_verified'
                                row.setdefault('transform_errors', []).append(str(exc))
                                padded = torch.nn.functional.pad(stored[index], (0, n-logical_n))
                                prepared = ttnn.from_torch(padded, device=mesh, dtype=ttnn.bfloat8_b,
                                                          layout=ttnn.TILE_LAYOUT, memory_config=weight_mem)
                            check = read(prepared).float()
                            assert torch.equal(check[:, :logical_n], stored[index]), 'Prequantized weight elements changed'
                            assert torch.count_nonzero(check[:, logical_n:]) == 0, 'Nonzero physical padding'
                            transformed.append(prepared)
                            transformations.append(method)
                            del check
                        sw = transformed
                        cached_geometry = identity
                    row.update(weight_elements_exact=True, weight_transformations=transformations,
                               activation_elements_exact=True, logical_k_unchanged=True,
                               observed_active_input_cores=active_input_cores)
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=geometry['block'], per_core_M=1, per_core_N=geometry['per_core_N'],
                        num_workers_per_dram_bank=geometry['readers'], use_multicore_pipeline=True)
                    record(row, lambda: projection(input_mem, sw, (pc, pc)))
                except (RuntimeError, AssertionError) as exc:
                    row.update(status='setup_error', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
                    report['rows'].append(row)
                    save()
                # No exception text or per-case metrics flood the console; exact details are in JSON.
            print(f'{key}: {sum(r["status"] == "measured" for r in report["rows"] if r["key"] == key)} measured rows; details in {args.output}', flush=True)
            del tx, sw, base_weights, stored, references, original
            gc.collect()
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    main()
