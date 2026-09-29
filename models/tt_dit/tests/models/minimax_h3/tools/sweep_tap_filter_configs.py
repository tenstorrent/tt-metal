# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Measure the depthwise tap filter's conv1d configurations and emit rows for ``utils/tap_filter_configs.py``.

Two steps, both on the device:

1. ``--record``: run the MiniMax-H3 audio decoder once and collect every ``(B, T_pad, C, K, stride)`` the tap
   filter is called with -- the production shapes, at the requested clip length and T-shard factor.
2. ``--sweep``: for each shape and each conv1d formulation that applies (full C, then chunk widths 128/64/32),
   find the smallest DRAM slice count the op runs with (explicit ``slice_config``, so the slicer never searches
   and a miss is just a RuntimeError), then time that count and a few larger ones and keep the fastest. The
   auto-slicer and the MAC fallback are timed alongside for reference.

The output is the two table blocks to paste into ``tap_filter_configs.py`` (formulation per ``(C, K, stride)``,
slice reference per ``(channels_run, K, stride)``), a markdown summary, and ``--json`` with every measurement.
Slice references are taken at the largest ``T_out`` swept for a key; the derived count for every other swept
length of that key is then run once to confirm it fits (``derive_num_slices`` rounds up, so it should).

Examples (repo root, venv active, TT_METAL_HOME/PYTHONPATH set, MINIMAX_H3_MODEL_PATH for --record)::

    python -m models.tt_dit.tests.models.minimax_h3.tools.sweep_tap_filter_configs --record --frames 207 \
        --sweep --json /tmp/tap_5s_f1.json
    python -m models.tt_dit.tests.models.minimax_h3.tools.sweep_tap_filter_configs --record --frames 603 \
        --mesh 4x8 --factor 8 --axis 1 --sweep --json /tmp/tap_15s_f8.json
    python -m models.tt_dit.tests.models.minimax_h3.tools.sweep_tap_filter_configs \
        --shapes 1x166x512x12x1 1x416x512x12x2 --sweep

Shapes are ``BxT_padxCxKxstride``. The sweep runs on device (0, 0) of the opened mesh; the fit depends on
per-core L1, so measure on the device class (architecture, grid) the table row is for, at the ``l1_small_size``
the pipeline opens its mesh with (default 65536, the value the MiniMax-H3 tests use).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
from loguru import logger

import ttnn
from models.tt_dit.layers import audio_ops, audio_resample
from models.tt_dit.utils.tap_filter_configs import (
    applicable_formulations,
    derive_num_slices,
    format_formulation_row,
    format_slice_row,
    slice_config_for,
    tap_device_key,
)

DEFAULT_MAX_SLICES = 16
DEFAULT_REPEAT = 5


# ------------------------------------------------------------------------------------------- recording


def record_decoder_shapes(mesh_device, *, num_latent_frames: int, factor: int, axis: int) -> Counter:
    """Run the H3 audio decoder once and count the distinct shapes the tap filter sees."""
    from safetensors.torch import load_file

    from models.tt_dit.models.audio_vae.minimax_h3.blockings_minimax_h3_audio import register_h3_audio_blockings
    from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
    from models.tt_dit.parallel.config import ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.minimax_h3.common import build_audio_decoder, load_config, weights_subdir

    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        sys.exit("--record needs MINIMAX_H3_MODEL_PATH pointing at a checkout with audio_vae/config.json")
    config = load_config(weights_dir)
    converted = convert_minimax_h3_audio_state_dict(
        load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
    )
    register_h3_audio_blockings()

    pc = None if factor <= 1 else ParallelFactor(factor=factor, mesh_axis=axis)
    ccl = None if pc is None else CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    decoder = build_audio_decoder(config, mesh_device, parallel_config=pc, ccl_manager=ccl)
    decoder.load_torch_state_dict(converted, strict=False)

    seen: Counter = Counter()
    real = audio_ops.depthwise_tap_filter

    def recording(x_BTC, taps, stride, **kwargs):
        seen[(int(x_BTC.shape[0]), int(x_BTC.shape[1]), int(x_BTC.shape[2]), len(taps), int(stride))] += 1
        return real(x_BTC, taps, stride, **kwargs)

    # The resampler imported the name, so patch its module namespace (and audio_ops for any other caller).
    audio_resample.depthwise_tap_filter = recording
    audio_ops.depthwise_tap_filter = recording
    try:
        torch.manual_seed(0)
        latents = torch.randn(2, config["latent_channels"], num_latent_frames) * 0.1
        decoder(latents)
    finally:
        audio_resample.depthwise_tap_filter = real
        audio_ops.depthwise_tap_filter = real
    return seen


# ------------------------------------------------------------------------------------------- sweeping


def _timed(fn, mesh_device, repeat: int) -> float:
    """Best-of-``repeat`` device-synchronised wall time of ``fn()``; the first call (compile) is excluded."""
    fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - t0)
    return best


def _run_formulation(x_dev, taps, stride, *, formulation, slice_config, mesh_device, cache, B, T_pad, C, K):
    """One conv1d formulation with an explicit (or None = auto) slice config, through audio_ops' own helpers."""
    T_out = (T_pad - K) // stride + 1
    if formulation == "mac":
        return audio_ops._depthwise_tap_mac(x_dev, taps, stride, T_out=T_out, dtype=ttnn.float32)
    channels = C if formulation == "direct" else formulation
    wkey = ("sweep_w", channels, stride, K, B, T_pad, str(slice_config))
    weight = cache.get(wkey)
    prepared = weight is not None
    if weight is None:
        weight = audio_ops._tap_weight(taps, channels, ttnn.float32, mesh_device)
    if "cc" not in cache:
        cache["cc"] = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
    conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    common = dict(
        B=B,
        T_pad=T_pad,
        T_out=T_out,
        K=K,
        stride=stride,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        conv_config=conv_config,
        compute_config=cache["cc"],
        slice_config=slice_config,
        cache=cache,
        wkey=wkey,
        prepared=prepared,
    )
    if formulation == "direct":
        return audio_ops._depthwise_tap_conv1d(x_dev, weight, C=C, **common)
    return audio_ops._depthwise_tap_conv1d_chunked(x_dev, weight, C=C, chunk=channels, **common)


def _reference(x, taps, stride):
    K, C = len(taps), x.shape[-1]
    weight = torch.tensor(taps, dtype=torch.float32).view(1, 1, K).expand(C, 1, K).contiguous()
    return torch.nn.functional.conv1d(x.transpose(1, 2), weight, stride=stride, groups=C).transpose(1, 2)


def sweep_shape(mesh_device, shape, *, max_slices: int, repeat: int, time_mac: bool) -> dict:
    """All measurements for one ``(B, T_pad, C, K, stride)``."""
    B, T_pad, C, K, stride = shape
    T_out = (T_pad - K) // stride + 1
    torch.manual_seed(0)
    x = torch.randn(B, T_pad, C, dtype=torch.float32)
    x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    taps = [float(t) for t in torch.randn(K)]
    expected = _reference(x, taps, stride)
    cache: dict = {}

    def runner(formulation, slice_config):
        return lambda: _run_formulation(
            x_dev,
            taps,
            stride,
            formulation=formulation,
            slice_config=slice_config,
            mesh_device=mesh_device,
            cache=cache,
            B=B,
            T_pad=T_pad,
            C=C,
            K=K,
        )

    def probe(formulation, slice_config):
        """(fits, max_abs_err) for one configuration -- one run; a RuntimeError is 'does not fit'."""
        try:
            out = runner(formulation, slice_config)()
        except RuntimeError as exc:
            return dict(fits=False, error=str(exc).splitlines()[0][:200])
        actual = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
        return dict(fits=True, max_abs_err=float((actual - expected).abs().max()))

    def time_all(configs: dict) -> dict:
        """Best-of-``repeat`` seconds per config, measured in two interleaved rounds (forward, then reversed) so
        a drift or an order effect cannot favour whichever config happened to run last."""
        best = {name: float("inf") for name in configs}
        order = list(configs)
        for round_order in (order, order[::-1]):
            for name in round_order:
                formulation, slice_config = configs[name]
                best[name] = min(best[name], _timed(runner(formulation, slice_config), mesh_device, repeat))
        return best

    result = dict(shape=dict(B=B, T_pad=T_pad, C=C, K=K, stride=stride, T_out=T_out), formulations={})
    for formulation in applicable_formulations(C):
        rows = {}
        n_min = None
        for n in range(1, max_slices + 1):
            r = probe(formulation, slice_config_for(n))
            rows[n] = r
            if r["fits"]:
                n_min = n
                break
        if n_min is not None:
            for n in sorted({n_min + 1, n_min + 2, 2 * n_min}):
                if n <= max_slices and n <= T_out:
                    rows[n] = probe(formulation, slice_config_for(n))
        auto = probe(formulation, None)
        configs = {str(n): (formulation, slice_config_for(n)) for n, r in rows.items() if r["fits"]}
        if auto["fits"]:
            configs["auto"] = (formulation, None)
        seconds = time_all(configs) if configs else {}
        for n, r in rows.items():
            if r["fits"]:
                r["seconds"] = seconds[str(n)]
        if auto["fits"]:
            auto["seconds"] = seconds["auto"]
        fitting = {n: r for n, r in rows.items() if r["fits"]}
        best_n = min(fitting, key=lambda n: fitting[n]["seconds"]) if fitting else None
        result["formulations"][str(formulation)] = dict(
            n_min=n_min,
            best_n=best_n,
            best_seconds=fitting[best_n]["seconds"] if best_n else None,
            auto=auto,
            by_num_slices={str(n): r for n, r in rows.items()},
        )
        logger.info(
            f"{shape}: {formulation!r}: n_min={n_min} best_n={best_n} "
            f"best={fitting[best_n]['seconds'] * 1e3 if best_n else float('nan'):.2f} ms "
            f"auto={'%.2f ms' % (auto['seconds'] * 1e3) if auto['fits'] else 'no fit'}"
        )
    if time_mac:
        mac = probe("mac", None)
        if mac["fits"]:
            mac["seconds"] = _timed(runner("mac", None), mesh_device, repeat)
        result["formulations"]["mac"] = mac
    ttnn.deallocate(x_dev)
    return result


# A slice row is recorded only when the explicit count beats auto-slicing by this margin at the reference length.
EXPLICIT_WIN_RATIO = 0.9


# ------------------------------------------------------------------------------------------- tables


def build_tables(results: list[dict]) -> tuple[dict, dict, list[str]]:
    """Formulation rows (fastest fitting formulation per (C, K, stride)) and slice references (per
    (channels, K, stride), at the largest T_out swept), plus derivation checks for the other lengths."""
    formulations: dict[tuple, object] = {}
    fastest: dict[tuple, float] = {}
    # (channels, K, stride) -> [(T_out, best explicit n, smallest fitting n, explicit seconds, auto seconds or None)]
    slice_candidates: dict[tuple, list[tuple[int, int, int, float, float | None]]] = {}
    for res in results:
        s = res["shape"]
        C, K, stride, T_out = s["C"], s["K"], s["stride"], s["T_out"]
        for name, meas in res["formulations"].items():
            if name == "mac" or (meas.get("best_n") is None and not meas.get("auto", {}).get("fits")):
                continue
            formulation = "direct" if name == "direct" else int(name)
            auto_s = meas["auto"]["seconds"] if meas.get("auto", {}).get("fits") else None
            explicit_s = meas.get("best_seconds")
            best_s = min(v for v in (explicit_s, auto_s) if v is not None)
            key = (C, K, stride)
            if best_s < fastest.get(key, float("inf")):
                fastest[key] = best_s
                formulations[key] = formulation
            if meas.get("best_n") is not None:
                channels = C if formulation == "direct" else formulation
                slice_candidates.setdefault((channels, K, stride), []).append(
                    (T_out, meas["best_n"], meas["n_min"], explicit_s, auto_s)
                )
    slices: dict[tuple, tuple[int, int]] = {}
    checks: list[str] = []
    for key, cands in slice_candidates.items():
        T_ref, n_ref, _, explicit_s, auto_s = max(cands, key=lambda c: c[0])
        if auto_s is not None and explicit_s > EXPLICIT_WIN_RATIO * auto_s:
            checks.append(
                f"{key}: no slice row -- explicit {explicit_s * 1e3:.2f} ms vs auto {auto_s * 1e3:.2f} ms at "
                f"T_out={T_ref}; conv1d's auto-slicing is kept"
            )
            continue
        slices[key] = (T_ref, n_ref)
        gain = f"{(1 - explicit_s / auto_s) * 100:.0f} % faster than auto" if auto_s else "auto does not fit"
        checks.append(f"{key}: slice row ({T_ref}, {n_ref}), {gain}")
        for T_out, n_best, n_min, _, _ in cands:
            derived = derive_num_slices(T_out, T_ref, n_ref)
            note = "fits" if derived >= n_min else "UNDER -- derived count smaller than the smallest count that fits"
            checks.append(f"    T_out={T_out}: smallest fitting {n_min}, fastest {n_best}, derived {derived}: {note}")
    return formulations, slices, checks


def print_tables(device_key, formulations, slices, checks, provenance: str) -> None:
    print("\n# ---- paste into models/tt_dit/utils/tap_filter_configs.py ----")
    print(f"# {provenance}")
    print(f"_FORMULATIONS[{device_key!r}] = {{")
    for (C, K, stride), f in sorted(formulations.items(), reverse=True):
        print(f"    {format_formulation_row(C, K, stride, f)}")
    print("}")
    print(f"_SLICES[{device_key!r}] = {{")
    for (channels, K, stride), (T_ref, n_ref) in sorted(slices.items(), reverse=True):
        print(f"    {format_slice_row(channels, K, stride, T_ref, n_ref)}")
    print("}")
    print("\n# slice rows: kept only where the explicit count beats auto-slicing; derived counts must be >= the")
    print("# measured minimum at every swept length:")
    for line in checks:
        print(f"#   {line}")


# ------------------------------------------------------------------------------------------- main


def _parse_shape(text: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(p) for p in text.lower().split("x"))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(f"shape must be BxT_padxCxKxstride, got {text!r}")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="run the H3 audio decoder and collect the tap shapes")
    ap.add_argument("--frames", type=int, default=207, help="latent frames for --record (207 = 5 s, 603 = 15 s)")
    ap.add_argument("--factor", type=int, default=1, help="audio T-shard factor for --record")
    ap.add_argument("--axis", type=int, default=1, help="mesh axis of the T shard for --record")
    ap.add_argument("--mesh", default="1x1", help="mesh shape to open, e.g. 1x1 or 4x8")
    ap.add_argument("--l1-small", type=int, default=65536, help="l1_small_size the mesh is opened with")
    ap.add_argument("--shapes", nargs="*", type=_parse_shape, default=[], help="extra shapes BxT_padxCxKxstride")
    ap.add_argument("--sweep", action="store_true", help="sweep formulations x slice counts for every shape")
    ap.add_argument("--max-slices", type=int, default=DEFAULT_MAX_SLICES)
    ap.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    ap.add_argument("--mac", action="store_true", help="also time the MAC fallback")
    ap.add_argument("--json", help="write every measurement here")
    ap.add_argument(
        "--no-table", action="store_true", help="clear this device class's table first (A/B against the probe chain)"
    )
    ap.add_argument(
        "--merge-json",
        nargs="*",
        default=[],
        help="no device: rebuild the tables from these --json outputs together (e.g. a 5 s and a 15 s sweep)",
    )
    args = ap.parse_args()

    if args.merge_json:
        merged, device_key, provenance = [], None, []
        for path in args.merge_json:
            with open(path) as fh:
                data = json.load(fh)
            key = tuple(data["device_key"])
            if device_key is not None and key != device_key:
                sys.exit(f"{path} is for device class {key}, the others for {device_key}; one table per class")
            device_key = key
            merged.extend(data["results"])
            provenance.append(f"{os.path.basename(path)} ({len(data['results'])} shapes, l1_small {data['l1_small']})")
        formulations, slices, checks = build_tables(merged)
        print_tables(device_key, formulations, slices, checks, "merged from " + ", ".join(provenance))
        return

    rows, cols = (int(v) for v in args.mesh.lower().split("x"))
    fabric = rows * cols > 1
    if fabric:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), l1_small_size=args.l1_small)
    try:
        device_key = tap_device_key(mesh_device)
        logger.info(f"device class {device_key}, mesh {rows}x{cols}, l1_small_size {args.l1_small}")
        if args.no_table:
            from models.tt_dit.utils.tap_filter_configs import clear_tap_configs

            clear_tap_configs(device_key)
            logger.info(f"table cleared for {device_key}: the filter probes as before")
        shapes: Counter = Counter({tuple(s): 0 for s in args.shapes})
        if args.record:
            seen = record_decoder_shapes(mesh_device, num_latent_frames=args.frames, factor=args.factor, axis=args.axis)
            logger.info(f"recorded {len(seen)} distinct tap shapes over {sum(seen.values())} calls:")
            for shape, count in sorted(seen.items(), key=lambda kv: (-kv[0][2], kv[0][1])):
                B, T_pad, C, K, stride = shape
                logger.info(f"  {B}x{T_pad}x{C}x{K}x{stride}  ({count} calls)")
            shapes.update(seen)
        if not shapes:
            sys.exit("nothing to do: give --shapes and/or --record")
        results = []
        if args.sweep:
            # The sweep runs single-device shapes: on a multi-device mesh the recorded per-device shape is the
            # same on every chip, so sweeping it on the mesh measures the replicated op, which is what production runs.
            for shape in sorted(shapes, key=lambda s: (-s[2], s[1])):
                results.append(
                    sweep_shape(mesh_device, shape, max_slices=args.max_slices, repeat=args.repeat, time_mac=args.mac)
                )
            formulations, slices, checks = build_tables(results)
            provenance = (
                f"swept {time.strftime('%Y-%m-%d')} on {os.uname().nodename}, mesh {rows}x{cols}, "
                f"l1_small_size {args.l1_small}, shapes: " + ", ".join("x".join(map(str, s)) for s in sorted(shapes))
            )
            print_tables(device_key, formulations, slices, checks, provenance)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(
                    dict(
                        device_key=list(device_key),
                        mesh=[rows, cols],
                        l1_small=args.l1_small,
                        shapes={"x".join(map(str, s)): n for s, n in shapes.items()},
                        results=results,
                    ),
                    fh,
                    indent=1,
                )
            logger.info(f"wrote {args.json}")
    finally:
        ttnn.close_mesh_device(mesh_device)
        if fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
