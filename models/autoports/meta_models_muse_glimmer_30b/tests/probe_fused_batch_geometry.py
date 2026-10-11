# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Primitive isolation of final decode geometry; this is NOT HF accuracy evidence.

Runs inherited FusedDecoder.decode_forward for B1..32 and both layer kinds.
Identity norms, supplied packed QKV, and identity finish avoid checkpoint/MLP
allocation. Native SDPA executes; its output is replaced by a prepared sentinel
to make concat ordering independently observable. Native head creation, RoPE,
cache update, SDPA, concat, and memory conversions are otherwise unchanged.

Two eager passes refresh persistent buffers, checking warm program reuse. This
does not capture a trace, benchmark performance, or validate model numerics.
Imports and --help are hardware-free; torch/TTNN/model imports occur in run().
Run serially under the hardware owner's normal visible-device environment:

    python -m models.autoports.meta_models_muse_glimmer_30b.tests.probe_fused_batch_geometry \
        --output /tmp/fused_batch_geometry.json
"""

import argparse
import gc
import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DECODER = ROOT / "tt/fused_decoder.py"
CONFIG = ROOT / "tests/assets/config.json"
PAGES_PER_USER = 8  # 32-byte page-table rows; at most 8 MiB for both BF16 caches.


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def device_only(ttnn):
    """Reject Python torch work and host conversions inside the inherited pass."""
    old = {name: getattr(ttnn, name) for name in ("from_torch", "to_torch", "as_tensor")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Host conversion inside geometry decode")

    def audit(frame, event, arg):
        module = frame.f_globals.get("__name__", "")
        if event == "call" and (module == "torch" or module.startswith("torch.")):
            raise AssertionError(f"Torch call inside geometry decode: {module}.{frame.f_code.co_name}")

    previous = sys.getprofile()
    try:
        for name in old:
            setattr(ttnn, name, forbidden)
        sys.setprofile(audit)
        yield
    finally:
        sys.setprofile(previous)
        for name, value in old.items():
            setattr(ttnn, name, value)


def run(args):
    import torch

    import ttnn
    from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import FusedDecoder

    class GeometryDecoder(FusedDecoder):
        def _norm(self, tensor, name=None, *, eps=None):
            return tensor

        def _linear(self, tensor, name):
            assert name == "qkv", name
            return self.packed

        def _finish(self, residual, normalized, attn):
            return attn

    cfg = json.loads(CONFIG.read_text())["text_config"]
    assert tuple(cfg[name] for name in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim")) == (
        6656,
        32,
        2,
        128,
    )
    report = {
        "status": "running",
        "evidence_type": "primitive_isolation_not_hf_accuracy",
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "decoder_sha256": sha256(DECODER),
        "probe_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(CONFIG),
        "torch_version": torch.__version__,
        "requested_batches": args.batches,
        "requested_kinds": args.kinds,
        "passes_per_case": args.passes,
        "assertion_policy": "exact equality on all logical geometry values and entire K/V caches",
        "isolation": {
            "norms": "identity",
            "qkv_projection": "prepared BF16 packed QKV",
            "finish": "identity",
            "sdpa": "native op executes; finite output checked; prepared sentinel feeds concat",
            "rope_angles": "distinct per-user quarter-turn patterns, not model position frequencies",
            "trace_capture": False,
        },
        "max_cache_bytes": max(args.batches) * PAGES_PER_USER * 2 * 32 * 128 * 2 * 2,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def to_device(value, *, dtype=None, layout=None):
        return ttnn.from_torch(
            value,
            device=mesh,
            dtype=ttnn.bfloat16 if dtype is None else dtype,
            layout=ttnn.TILE_LAYOUT if layout is None else layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def read(tensor):
        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    def copy(value, target):
        host = ttnn.from_torch(value, dtype=target.dtype, layout=target.layout)
        ttnn.copy_host_to_device_tensor(host, target)

    def exact(name, got, expected):
        assert tuple(got.shape) == tuple(expected.shape), (name, got.shape, expected.shape)
        assert torch.isfinite(got).all(), f"{name}: nonfinite value"
        if not torch.equal(got, expected):
            count = (got != expected).sum().item()
            error = (got.float() - expected.float()).abs().max().item()
            raise AssertionError(f"{name}: {count} unequal elements; maximum absolute error {error}")

    def shape(name, tensor, logical, padded):
        actual = {"logical": list(tensor.shape), "padded": list(tensor.padded_shape)}
        assert actual == {"logical": logical, "padded": padded}, (name, actual, logical, padded)
        return actual

    def cores(tensor):
        spec = tensor.memory_config().shard_spec
        assert list(spec.shape) == [32, 128], spec
        assert spec.orientation == ttnn.ShardOrientation.ROW_MAJOR, spec
        found = {(x, y) for y in range(8) for x in range(8) if spec.grid.contains(ttnn.CoreCoord(x, y))}
        assert len(found) == spec.grid.num_cores(), spec.grid
        return found

    def case(batch, kind):
        rope = kind == "rope"
        partial = batch > 8 and batch % 8 != 0
        generator = torch.Generator().manual_seed(args.seed + batch)

        def random_values(dims):
            return (torch.randint(-64, 65, dims, generator=generator).float() / 32).bfloat16()

        layer = GeometryDecoder()
        layer.use_rope = rope
        layer.window = cfg["sliding_window"] if rope else None
        layer.attention_scale = torch.tensor(128**-0.5 * cfg["qk_scale_factor"], dtype=torch.float32).item()
        layer.ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        layer.sdpa = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )
        cache_shape = (batch * PAGES_PER_USER, 2, 32, 128)
        expected_cache = [random_values(cache_shape), random_values(cache_shape)]
        cache = tuple(to_device(value) for value in expected_cache)
        x = to_device(torch.zeros(1, 1, batch, 6656, dtype=torch.bfloat16))
        persistent = {}
        rows = []
        for step in range(args.passes):
            q = random_values((1, batch, 32, 128))
            k = random_values((1, batch, 2, 128))
            v = random_values((1, batch, 2, 128))
            packed = torch.cat([tensor.reshape(1, 1, batch, -1) for tensor in (q, k, v)], dim=-1)
            # Base-four user codes give every user a different angle vector.
            users = (torch.arange(batch) + step * 37) % 64
            powers = 4 ** (torch.arange(64) % 3)
            quadrants = users[:, None] // powers[None, :] % 4
            cos_half = torch.tensor([1, 0, -1, 0], dtype=torch.bfloat16)[quadrants]
            sin_half = torch.tensor([0, 1, 0, -1], dtype=torch.bfloat16)[quadrants]
            cos = torch.cat((cos_half, cos_half), dim=-1)[None, :, None, :]
            sin = torch.cat((sin_half, sin_half), dim=-1)[None, :, None, :]
            assert len({tuple(row) for row in quadrants.tolist()}) == batch
            # Include tile/face/page boundaries and distinct positions within each 32-user batch.
            boundaries = torch.tensor(
                [
                    0,
                    1,
                    15,
                    16,
                    31,
                    32,
                    33,
                    63,
                    64,
                    95,
                    96,
                    127,
                    128,
                    159,
                    160,
                    191,
                    192,
                    223,
                    224,
                    255,
                    2,
                    14,
                    17,
                    30,
                    34,
                    62,
                    65,
                    94,
                    97,
                    126,
                    129,
                    254,
                ],
                dtype=torch.int32,
            )
            pos = boundaries.roll(step * 7)[:batch].contiguous()
            pt = torch.randperm(batch * PAGES_PER_USER, generator=generator).reshape(batch, PAGES_PER_USER).int()
            sentinel = random_values((1, batch, 32, 128))
            sentinel[0, :, :, 0] = torch.arange(batch)[:, None]
            sentinel[0, :, :, 1] = torch.arange(32)[None, :]
            values = {"packed": packed, "cos": cos, "sin": sin, "pos": pos, "pt": pt, "sentinel": sentinel}
            for name, value in values.items():
                if name not in persistent:
                    integer = name in ("pos", "pt")
                    persistent[name] = to_device(
                        value,
                        dtype=ttnn.int32 if integer else ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
                    )
                else:
                    copy(value, persistent[name])
            layer.packed = persistent["packed"]
            captured = {"rope_calls": []}
            native_rope = ttnn.experimental.rotary_embedding_hf
            native_cache = ttnn.experimental.paged_fused_update_cache
            native_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode
            native_concat = ttnn.experimental.nlp_concat_heads_decode

            def spy_rope(tensor, cosine, sine, **kwargs):
                result = native_rope(tensor, cosine, sine, **kwargs)
                captured["rope_calls"].append((tensor, cosine, sine, result, kwargs.get("is_decode_mode", False)))
                return result

            def spy_cache(cache_k, key, cache_v, value, **kwargs):
                assert "k" not in captured, "Unexpected additional cache update"
                captured.update(k=key, v=value)
                return native_cache(cache_k, key, cache_v, value, **kwargs)

            def spy_sdpa(query, *operands, **kwargs):
                assert "q" not in captured, "Unexpected additional SDPA"
                captured["q"] = query
                captured["native_sdpa"] = native_sdpa(query, *operands, **kwargs)
                return persistent["sentinel"]

            def spy_concat(tensor, **kwargs):
                assert "concat_input" not in captured, "Unexpected additional concat"
                captured["concat_input"] = tensor
                captured["concat_subgrid"] = kwargs.get("sub_core_grids")
                captured["concat_output"] = native_concat(tensor, **kwargs)
                return captured["concat_output"]

            with (
                patch.object(ttnn.experimental, "rotary_embedding_hf", spy_rope),
                patch.object(ttnn.experimental, "paged_fused_update_cache", spy_cache),
                patch.object(ttnn.transformer, "paged_scaled_dot_product_attention_decode", spy_sdpa),
                patch.object(ttnn.experimental, "nlp_concat_heads_decode", spy_concat),
            ):
                with device_only(ttnn):
                    output = layer.decode_forward(
                        x,
                        kv_cache=cache,
                        page_table=persistent["pt"],
                        current_pos=persistent["pos"],
                        rope_cos=persistent["cos"] if rope else None,
                        rope_sin=persistent["sin"] if rope else None,
                    )
            ttnn.synchronize_device(mesh)
            shapes = {
                "q": shape("q", captured["q"], [1, batch, 32, 128], [1, batch, 32, 128]),
                "k": shape("k", captured["k"], [1, batch, 2, 128], [1, batch, 32, 128]),
                "v": shape("v", captured["v"], [1, batch, 2, 128], [1, batch, 32, 128]),
                "output": shape("output", output, [1, 1, batch, 4096], [1, 1, 32, 4096]),
            }
            assert len(captured["rope_calls"]) == (2 if rope else 0)
            for head_count, (tensor, cosine, sine, result, decode_mode) in zip((32, 2), captured["rope_calls"]):
                assert decode_mode == (not partial)
                logical = [1, head_count, batch, 128] if partial else [1, batch, head_count, 128]
                padded = [1, head_count, 32, 128] if partial else [1, batch, 32, 128]
                shape("rope input", tensor, logical, padded)
                shape("rope output", result, logical, padded)
                trig_logical = [1, 1, batch, 128] if partial else [1, batch, 1, 128]
                trig_padded = [1, 1, 32, 128] if partial else [1, batch, 32, 128]
                shape("cos", cosine, trig_logical, trig_padded)
                shape("sin", sine, trig_logical, trig_padded)
            upper = {(i % 8, i // 8) for i in range(batch)}
            lower = {(i % 8, 4 + i // 8) for i in range(batch)}
            assert cores(captured["k"]) == lower
            assert cores(captured["v"]) == upper
            assert cores(captured["concat_input"]) == upper
            assert lower.isdisjoint(upper)
            q_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if rope and not partial else ttnn.TensorMemoryLayout.INTERLEAVED
            )
            assert captured["q"].memory_config().memory_layout == q_layout
            subgrid = captured["concat_subgrid"]
            if partial:
                assert subgrid is not None and subgrid.num_cores() == 32
                assert all(subgrid.contains(ttnn.CoreCoord(xc, yc)) for yc in range(4) for xc in range(8))
            else:
                assert subgrid is None

            def rotated(tensor):
                half = torch.cat((-tensor[..., 64:], tensor[..., :64]), dim=-1)
                return (tensor * cos + half * sin).bfloat16() if rope else tensor

            wanted_q, wanted_k = rotated(q), rotated(k)
            exact("per-user Q", read(captured["q"]), wanted_q)
            exact("per-user K", read(captured["k"]), wanted_k)
            exact("per-user V", read(captured["v"]), v)
            for user in range(batch):
                page = pt[user, pos[user].item() // 32].item()
                offset = pos[user].item() % 32
                expected_cache[0][page, :, offset, :] = wanted_k[0, user]
                expected_cache[1][page, :, offset, :] = v[0, user]
            exact("entire K cache including untouched elements", read(cache[0]), expected_cache[0])
            exact("entire V cache including untouched elements", read(cache[1]), expected_cache[1])
            exact("concat input sentinel", read(captured["concat_input"]), sentinel)
            exact("concat head/user ordering", read(output), sentinel.reshape(1, 1, batch, 4096))
            shape("native SDPA", captured["native_sdpa"], [1, batch, 32, 128], [1, batch, 32, 128])
            assert torch.isfinite(read(captured["native_sdpa"])).all(), "Native SDPA returned nonfinite values"
            rows.append(
                {
                    "pass": step,
                    "status": "pass",
                    "positions": pos.tolist(),
                    "shapes": shapes,
                    "addressed_cache_rows": [
                        [pt[u, pos[u].item() // 32].item(), pos[u].item() % 32] for u in range(batch)
                    ],
                    "per_user_qkv_exact": [True] * batch,
                    "entire_kv_cache_exact": True,
                    "concat_exact": True,
                    "native_sdpa_finite": True,
                }
            )
            print(f"PASS batch={batch} kind={kind} pass={step} partial_row={partial}", flush=True)
            del captured, output
        return {"batch": batch, "kind": kind, "partial_row": partial, "passes": rows}

    save()
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
        report["arch"] = str(mesh.arch())
        for batch in args.batches:
            for kind in args.kinds:
                report["active_case"] = {"batch": batch, "kind": kind}
                save()
                report["cases"].append(case(batch, kind))
                gc.collect()
                save()
        assert sha256(DECODER) == report["decoder_sha256"], "Decoder source changed during the probe"
        report.pop("active_case", None)
        report["status"] = "pass"
        print("ALL_GEOMETRY_PASS", flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["utc_finished"] = datetime.now(timezone.utc).isoformat()
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", nargs="+", type=int, default=list(range(1, 33)))
    parser.add_argument("--kinds", nargs="+", choices=("rope", "nope"), default=["rope", "nope"])
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(batch < 1 or batch > 32 for batch in args.batches):
        parser.error("Every batch must be in 1..32")
    if args.passes < 1:
        parser.error("--passes must be positive")
    run(args)


if __name__ == "__main__":
    main()
