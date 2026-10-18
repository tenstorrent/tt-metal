"""Local-head/state ownership comparisons against optimized single-chip attention."""

import argparse
import os, hashlib, json, math
from pathlib import Path
import torch, ttnn
from transformers import AutoConfig
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_components import CHECKPOINT
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import no_runtime_host_conversion
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import (
    OptimizedGatedDeltaNet,
    OptimizedQSA,
    OptimizedHyperConnection,
    _configure_dram,
    _prefill_linear,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, ParallelGatedDeltaNet, ParallelQSA


class MatchedPrecisionQSA(OptimizedQSA):
    """Full-width single-chip control with the mesh QSA precision policy."""

    projection_fidelity = ttnn.MathFidelity.LoFi
    packed_dtype = ttnn.bfloat4_b

    @classmethod
    def from_state_dict(cls, state, **kwargs):
        obj = super().from_state_dict(state, **kwargs)
        qg = state["q_proj.weight"].reshape(24, 512, 2560)
        packed = torch.cat(
            (
                qg[:, :256].reshape(6144, 2560),
                state["k_proj.weight"],
                state["v_proj.weight"],
                qg[:, 256:].reshape(6144, 2560),
                state["indexer.index_qk_proj.weight"],
            ),
            dim=0,
        ).T.contiguous()
        assert tuple(packed.shape) == (2560, 13952)
        obj.packed_projection = ttnn.from_torch(
            packed,
            dtype=cls.packed_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=obj.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        obj.wo = ttnn.from_torch(
            state["o_proj.weight"].T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=obj.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        obj.projection_computes = {w.buffer_address(): obj.projection_compute for w in (obj.packed_projection, obj.wo)}
        # Refresh native single-device DRAM copies; no mesh layout or collectives.
        _configure_dram(obj, ("packed_projection", "wo"), cores=(10, 8), blocks=(8, 24), readers=(2, 2))
        return obj

    def _linear(self, x, weight):
        if weight is not self.wo or x.shape[-2] != 1 or math.prod(tuple(x.shape)[:-2]) != 1:
            return super()._linear(x, weight)
        shape = tuple(x.shape)
        w, memory, program = self.dram_projections[weight.buffer_address()]
        payload = ttnn.typecast(ttnn.reshape(x, (1, 1, 1, shape[-1])), ttnn.bfloat8_b)
        out = ttnn.linear(
            ttnn.to_memory_config(payload, memory),
            w,
            program_config=program,
            compute_kernel_config=self.projection_compute,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        if self.route_sequence > 1:
            out = ttnn.typecast(out, ttnn.bfloat16)
        return ttnn.reshape(out, (*shape[:-1], weight.shape[-1]))


class HybridPrecisionQSA(MatchedPrecisionQSA):
    """Independent single control: BF8 prefill and explicit BF8-to-BF4 decode."""

    packed_dtype = ttnn.bfloat8_b

    @classmethod
    def from_state_dict(cls, state, **kwargs):
        obj = super().from_state_dict(state, **kwargs)
        obj.decode_projection = ttnn.typecast(obj.packed_projection, ttnn.bfloat4_b)
        obj.projection_computes[obj.decode_projection.buffer_address()] = obj.projection_compute
        obj.prefill_projection_compute = ttnn.init_device_compute_kernel_config(
            obj.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        _configure_dram(
            obj,
            ("packed_projection", "decode_projection", "wo"),
            cores=(10, 10, 8),
            blocks=(8, 8, 24),
            readers=(2, 2, 2),
        )
        obj._decode_phase = False
        return obj

    def prefill_forward(self, hidden_states, **kwargs):
        self._decode_phase = False
        return super().prefill_forward(hidden_states, **kwargs)

    def decode_forward(self, hidden_states, **kwargs):
        self._decode_phase = True
        return super().decode_forward(hidden_states, **kwargs)

    def _linear(self, x, weight):
        if weight is self.packed_projection:
            if self._decode_phase:
                return super()._linear(x, self.decode_projection)
            if x.shape[-2] > 1:
                return _prefill_linear(x, weight, self.prefill_projection_compute)
        return super()._linear(x, weight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["single", "mesh"], required=True)
    ap.add_argument("--part", choices=["gdn", "qsa"], required=True)
    ap.add_argument("--length", type=int, default=33)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tag", default="attention")
    ap.add_argument("--reference-tag")
    ap.add_argument("--rank3", action="store_true", help="GDN rank-three input contract")
    ap.add_argument("--recorded-tag", help="QSA B1: derive identical attention inputs through optimized HC on single")
    ap.add_argument("--matched-qsa-precision", action="store_true", help="Compare mesh with a BF4/LoFi single control")
    ap.add_argument(
        "--hybrid-qsa-precision", action="store_true", help="Matched BF8 prefill / BF8-to-BF4 decode control"
    )
    ap.add_argument("--decode-step", action="store_true", help="Append one recorded QSA token and verify output/cache")
    ap.add_argument("--original-reference-tag", help="Preserved original single control for precision attribution")
    a = ap.parse_args()
    if a.hybrid_qsa_precision:
        a.matched_qsa_precision = True
    if a.recorded_tag and (a.part != "qsa" or a.batch != 1 or a.rank3):
        ap.error("--recorded-tag requires --part qsa --batch 1 without --rank3")
    if a.recorded_tag and a.tag == "attention":
        ap.error("Use an explicit --tag for recorded attention references")
    if a.decode_step and not a.recorded_tag:
        ap.error("--decode-step requires recorded QSA B1 inputs")
    if a.matched_qsa_precision and (not a.recorded_tag or not a.original_reference_tag):
        ap.error("--matched-qsa-precision requires --recorded-tag and --original-reference-tag")
    if a.original_reference_tag and not a.matched_qsa_precision:
        ap.error("--original-reference-tag requires --matched-qsa-precision")
    if a.matched_qsa_precision and (a.tag == a.original_reference_tag or a.reference_tag == a.original_reference_tag):
        ap.error("Use a separate tag/reference-tag for the matched control; preserve the original artifacts")
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    layer = 0 if a.part == "gdn" else 3
    prefix = f"model.language_model.layers.{layer}." + ("linear_attn." if a.part == "gdn" else "self_attn.")
    state = CheckpointState(CHECKPOINT, prefix)
    cache = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip")
    cache.mkdir(exist_ok=True)
    dest = Path(__file__).parents[1] / os.environ.get("QWEN38_STAGE_DIR", "doc/multichip_decoder")
    original_path = cache / f"{a.original_reference_tag}_single_{a.part}_b{a.batch}_s{a.length}.pt"
    if a.matched_qsa_precision:
        if not original_path.is_file():
            raise FileNotFoundError(f"Run the original single control first: {original_path}")
        if a.impl == "mesh":
            control_path = dest / f"{a.reference_tag or a.tag}_single_{a.part}_b{a.batch}_s{a.length}.json"
            if not control_path.exists():
                control_path = Path(__file__).parents[1] / "doc/multichip_decoder" / control_path.name
            control = json.loads(control_path.read_text())
            if (
                not control["args"].get("matched_qsa_precision")
                or control["args"].get("original_reference_tag") != a.original_reference_tag
                or bool(control["args"].get("hybrid_qsa_precision")) != a.hybrid_qsa_precision
                or bool(control["args"].get("decode_step")) != a.decode_step
            ):
                raise ValueError("Mesh reference must be the matched single control for the same original tag")
    fixture_info = None
    prepared_input = None
    input_tag = (a.reference_tag or a.tag) if a.impl == "mesh" else a.tag
    if a.matched_qsa_precision:
        input_tag = a.original_reference_tag
    input_path = cache / f"{input_tag}_input_{a.part}_b{a.batch}_s{a.length}.pt"
    if a.recorded_tag:
        fixture_path = cache / a.recorded_tag / f"layer{layer}.pt"
        fixture_info = dict(
            path=str(fixture_path),
            sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            hc_source_sha256=hashlib.sha256(
                (Path(__file__).parents[1] / "tt/optimized_decoder.py").read_bytes()
            ).hexdigest(),
            hc_base_source_sha256=hashlib.sha256(
                (Path(__file__).parents[1] / "tt/fused_decoder.py").read_bytes()
            ).hexdigest(),
            length=a.length,
            batch=a.batch,
            hc_chunk_size=512,
        )
        if a.decode_step:
            fixture_info.update(decode_step=True, decode_position=a.length)
        if a.decode_step and a.impl == "single" and not a.matched_qsa_precision and input_path.exists():
            previous = torch.load(input_path, map_location="cpu", weights_only=True)
            if previous["fixture"] != fixture_info:
                raise ValueError("Use a fresh original tag for the recorded prefill-plus-decode fixture")
        if a.impl == "mesh" or a.matched_qsa_precision:
            prepared_input = torch.load(input_path, map_location="cpu", weights_only=True)
            if prepared_input["fixture"] != fixture_info:
                raise ValueError("Recorded attention input provenance differs; rerun the single control")
            if a.decode_step and "decode" not in prepared_input:
                raise ValueError("Original attention fixture lacks the appended decode input")
        if a.impl == "mesh" and a.matched_qsa_precision and control["fixture"] != fixture_info:
            raise ValueError("Matched single control used a different recorded attention fixture")
    if a.impl == "mesh":
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 4 if a.impl == "mesh" else 1), trace_region_size=0)
    try:
        if a.impl == "mesh":
            ops = MeshOps(m)
            model = (
                ParallelGatedDeltaNet(state, cfg, ops, a.batch, a.batch + 1)
                if a.part == "gdn"
                else ParallelQSA.build(state, cfg, ops, layer, a.batch, a.batch + 1)
            )
            if a.hybrid_qsa_precision:
                assert model.packed_projection.dtype == ttnn.bfloat8_b, "Mesh must use the production hybrid policy"
                assert getattr(model, "decode_projection", None) is not None, "Missing hybrid decode weight"
                assert model.decode_projection.dtype == ttnn.bfloat4_b
        else:
            control_class = (
                OptimizedGatedDeltaNet
                if a.part == "gdn"
                else (
                    HybridPrecisionQSA
                    if a.hybrid_qsa_precision
                    else MatchedPrecisionQSA if a.matched_qsa_precision else OptimizedQSA
                )
            )
            model = control_class.from_state_dict(
                state, hf_config=cfg, layer_idx=layer, mesh_device=m, batch_size=a.batch, max_requests=a.batch + 1
            )

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return (
                ops.upload(x, dtype, layout)
                if a.impl == "mesh"
                else ttnn.from_torch(x, dtype=dtype, layout=layout, device=m)
            )

        rng = torch.Generator().manual_seed(322)
        input_host = torch.randn(1, a.batch, a.length, 2560, generator=rng).bfloat16() * 0.05
        decode_host = None
        if a.recorded_tag:
            if a.impl == "single" and not a.matched_qsa_precision:
                fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
                residuals = torch.cat(fixture["inputs"], dim=1).unsqueeze(0)
                if residuals.shape[2] < a.length + int(a.decode_step):
                    raise ValueError("Recorded fixture is shorter than the requested attention sequence")
                hc = OptimizedHyperConnection.from_state_dict(
                    CheckpointState(CHECKPOINT, f"model.language_model.layers.{layer}."),
                    prefix="attn_hyper_connection",
                    hf_config=cfg,
                    mesh_device=m,
                )
                chunks = []
                for start in range(0, a.length, 512):
                    residual = upload(residuals[:, :, start : min(start + 512, a.length), :].contiguous())
                    with no_runtime_host_conversion():
                        mixed, inject = hc.read(residual)
                    chunks.append(ttnn.to_torch(mixed))
                    del mixed, inject, residual
                input_host = torch.cat(chunks, dim=2)
                saved_input = dict(input=input_host, fixture=fixture_info)
                if a.decode_step:
                    residual = upload(residuals[:, :, a.length : a.length + 1, :].contiguous())
                    with no_runtime_host_conversion():
                        mixed, inject = hc.read(residual)
                    decode_host = ttnn.to_torch(mixed)
                    saved_input["decode"] = decode_host
                    del mixed, inject, residual
                torch.save(saved_input, input_path)
                del hc, chunks
            else:
                input_host = prepared_input["input"]
                decode_host = prepared_input.get("decode")
            assert tuple(input_host.shape) == (1, a.batch, a.length, 2560)
            if a.decode_step:
                assert tuple(decode_host.shape) == (1, a.batch, 1, 2560)
        x = upload(input_host)
        p = torch.zeros(a.batch, 32, dtype=torch.int32)
        s = p.clone()
        s[:, 0] = torch.arange(1, a.batch + 1)
        pd = upload(p, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        sd = upload(s, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        page = torch.randperm(a.batch * cfg.max_position_embeddings // 32, generator=rng).reshape(a.batch, -1).int()
        table = upload(page, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kw = (
            dict(current_positions=pd, slots=sd)
            if a.part == "gdn"
            else dict(current_pos=pd, page_table=table, slots=sd)
        )
        if a.rank3:
            assert a.part == "gdn"
            x = ttnn.reshape(x, (a.batch, a.length, 2560))
        with no_runtime_host_conversion():
            out = model.prefill_forward(x, **kw)
        local_shape, local_dtype = list(out.shape), str(out.dtype)
        assert out.dtype == (
            ttnn.bfloat8_b
            if (a.impl == "mesh" or a.matched_qsa_precision) and a.batch == a.length == 1
            else ttnn.bfloat16
        )
        assert out.shape[-1] == (640 if a.impl == "mesh" else 2560)
        if a.impl == "mesh":
            out = ops.gather(out)
        outputs = {"prefill": ttnn.to_torch(ttnn.get_device_tensors(out)[0])}
        del out
        decode_local_shape, decode_local_dtype = None, None
        if a.part == "gdn":
            states = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(model.recurrent_bank)]
            convs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(model.conv_bank)]
            outputs["state"] = torch.cat(states, dim=1) if a.impl == "mesh" else states[0]
            if a.impl == "mesh":
                outputs["conv"] = torch.cat(
                    [
                        torch.cat([v[..., start:stop] for v in convs], -1)
                        for start, stop in [(0, 512), (512, 1024), (1024, 2560)]
                    ],
                    -1,
                )
            else:
                outputs["conv"] = convs[0]
        else:

            def qsa_snapshot(length, *, check_replicas=False):
                positions = torch.arange(length)
                ids = page[:, positions // 32] * 32 + positions % 32
                samples = []
                # FP8 direct host conversion is unsupported; decode only in this
                # inspection harness, after the measured model call. Cache stays FP8.
                readable_cache = ttnn.typecast(model.packed_cache, ttnn.bfloat16) if model.packed_cache.dtype == ttnn.fp8_e4m3 else model.packed_cache
                for t in ttnn.get_device_tensors(readable_cache):
                    full = ttnn.to_torch(t).reshape(-1, 512)
                    samples.append(full[ids] if a.impl == "mesh" else full.reshape(-1, 2, 512)[ids])
                    del full
                if a.impl == "mesh":
                    assert torch.equal(samples[0], samples[1])
                    assert torch.equal(samples[2], samples[3])
                    packed = torch.stack((samples[0], samples[2]), dim=-2)
                else:
                    packed = samples[0]
                banks = ttnn.get_device_tensors(model.pooled_bank)
                pooled = ttnn.to_torch(banks[0])
                if check_replicas:
                    for bank in banks[1:]:
                        assert torch.equal(pooled, ttnn.to_torch(bank)), "Replicated pooled index state differs"
                return packed, pooled

            outputs["cache"], outputs["pooled"] = qsa_snapshot(a.length)
            if a.decode_step:
                dx = upload(decode_host)
                offset = torch.zeros_like(p)
                offset[:, 0] = a.length
                offset_device = upload(offset, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                with no_runtime_host_conversion():
                    pd = ttnn.add(pd, offset_device)
                    kw["current_pos"] = pd
                    out = model.decode_forward(dx, **kw)
                decode_local_shape, decode_local_dtype = list(out.shape), str(out.dtype)
                assert tuple(out.shape) == (1, 1, 1, 640 if a.impl == "mesh" else 2560)
                assert out.dtype == (ttnn.bfloat8_b if a.impl == "mesh" or a.matched_qsa_precision else ttnn.bfloat16)
                if a.impl == "mesh":
                    out = ops.gather(out)
                outputs["decode"] = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
                del out, dx, offset_device
                outputs["decode_cache"], outputs["decode_pooled"] = qsa_snapshot(a.length + 1, check_replicas=True)
                assert torch.equal(outputs["decode_cache"][:, : a.length], outputs["cache"]), "Decode changed prefix KV"
                # Compare the appended row separately so a long prefix cannot hide a bad write.
                outputs["decode_cache_row"] = outputs["decode_cache"][:, -1:]
                complete = (a.length + 1) // 4
                if complete:
                    outputs["decode_pooled_live"] = outputs["decode_pooled"][1:2, :, :complete, :]
                if complete == a.length // 4:
                    assert torch.equal(
                        outputs["decode_pooled"], outputs["pooled"]
                    ), "Incomplete group changed pooled state"
        path = cache / f"{a.tag}_{a.impl}_{a.part}_b{a.batch}_s{a.length}.pt"
        torch.save(outputs, path)
        comparisons = {}
        metrics = {}
        original = None
        if a.matched_qsa_precision:
            original = torch.load(original_path, map_location="cpu", weights_only=True)
            comparisons["matched_single_vs_original_single" if a.impl == "single" else "mesh_vs_original_single"] = {
                name: pcc_metrics(original[name], value) for name, value in outputs.items()
            }
        if a.impl == "mesh":
            refs = torch.load(
                cache / f"{a.reference_tag or a.tag}_single_{a.part}_b{a.batch}_s{a.length}.pt", weights_only=True
            )
            metrics = {name: pcc_metrics(refs[name], value) for name, value in outputs.items()}
            comparisons["mesh_vs_matched_single" if a.matched_qsa_precision else "mesh_vs_original_single"] = metrics
            if original is not None:
                comparisons["matched_single_vs_original_single"] = {
                    name: pcc_metrics(original[name], value) for name, value in refs.items()
                }
            print(json.dumps(metrics), flush=True)
        source = (
            Path(__file__).parents[1] / "tt" / ("multichip_decoder.py" if a.impl == "mesh" else "optimized_decoder.py")
        )
        snapshot_stem = f"{a.tag}_{a.impl}_{a.part}_b{a.batch}_s{a.length}"
        (dest / f"{snapshot_stem}_runner.py.txt").write_bytes(Path(__file__).read_bytes())
        (dest / f"{snapshot_stem}_source.py.txt").write_bytes(source.read_bytes())
        report_name = f"{a.tag}_{a.part}_b{a.batch}_s{a.length}.json" if a.impl == "mesh" else f"{snapshot_stem}.json"
        (dest / report_name).write_text(
            json.dumps(
                dict(
                    status=(
                        ("passed" if min(v["pcc"] for v in metrics.values()) >= 0.995 else "failed_pcc")
                        if a.impl == "mesh"
                        else "control_saved"
                    ),
                    metrics=metrics,
                    comparisons=comparisons,
                    args=vars(a),
                    local_shape=local_shape,
                    local_dtype=local_dtype,
                    decode_local_shape=decode_local_shape,
                    decode_local_dtype=decode_local_dtype,
                    fixture=fixture_info,
                    prepared_input=str(input_path) if a.recorded_tag else None,
                    outputs=str(path),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    matched_precision_policy=(
                        dict(
                            packed_weight=(
                                "BF8 directly from checkpoint"
                                if a.hybrid_qsa_precision
                                else "BF4 directly from checkpoint"
                            ),
                            decode_weight=(
                                "BF4 typecast from BF8, explicit decode only"
                                if a.hybrid_qsa_precision
                                else "same packed weight"
                            ),
                            wo_weight="BF8 directly from checkpoint",
                            projection_fidelity=(
                                "HiFi2 multi-token prefill; LoFi singleton prefill/decode/WO"
                                if a.hybrid_qsa_precision
                                else "LoFi"
                            ),
                            singleton_wo_input="BF8",
                            singleton_wo_output="BF8",
                            concatenated_tail="BF16",
                            single_geometry="native full-width optimized prefill/DRAM matmul",
                        )
                        if a.matched_qsa_precision
                        else None
                    ),
                ),
                indent=2,
            )
        )
        if comparisons:
            print(json.dumps(dict(comparisons=comparisons)), flush=True)
        if a.impl == "mesh":
            assert min(v["pcc"] for v in metrics.values()) >= 0.995
        print("ATTENTION_PASS", a.impl, a.part, a.batch, a.length, flush=True)
    finally:
        ttnn.close_mesh_device(m)


if __name__ == "__main__":
    main()
