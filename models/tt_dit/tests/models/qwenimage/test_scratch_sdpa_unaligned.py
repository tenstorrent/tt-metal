# Scratch: do joint and ring joint SDPA handle spatial lengths that are not tile multiples?

import time
from collections.abc import Callable
from typing import Any

import pytest
import torch

from loguru import logger

import ttnn
from models.tt_dit.blocks.attention import Attention
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params, line_params_req_exact_devices

HEADS = 24
HEAD_DIM = 128
PROMPT_LENGTH = 128


def _inputs(n: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    torch.manual_seed(0)
    spatial = [torch.randn(1, HEADS, n, HEAD_DIM).bfloat16().float() for _ in range(3)]
    prompt = [torch.randn(1, HEADS, PROMPT_LENGTH, HEAD_DIM).bfloat16().float() for _ in range(3)]
    return spatial, prompt


def _reference(spatial: list[torch.Tensor], prompt: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = (torch.cat([s, p], dim=2) for s, p in zip(spatial, prompt, strict=True))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    n = spatial[0].shape[2]
    return out[:, :, :n], out[:, :, n:]


def _program_config(device: ttnn.MeshDevice, *, exp_approx_mode: bool = False) -> ttnn.SDPAProgramConfig:
    grid = device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y - 1),
        q_chunk_size=128,
        k_chunk_size=512,
        exp_approx_mode=exp_approx_mode,
    )


_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("n", [4096, 7056, 6534])
@pytest.mark.parametrize("pad_value", [0.0, float("nan")], ids=["zero", "nan"])
def test_joint(*, mesh_device: ttnn.MeshDevice, n: int, pad_value: float) -> None:
    spatial, prompt = _inputs(n)
    ref_spatial, ref_prompt = _reference(spatial, prompt)

    # Fill the tile padding with NaN, to see whether it leaks into the result.
    tt = [tensor.from_torch(x, device=mesh_device, pad_value=pad_value) for x in spatial + prompt]
    assert tt[0].shape[2] == n

    out_spatial, out_prompt = ttnn.transformer.joint_scaled_dot_product_attention(
        *tt,
        joint_strategy="rear",
        program_config=_program_config(mesh_device),
        compute_kernel_config=_COMPUTE_CONFIG,
    )

    assert out_spatial.shape[2] == n
    assert_quality(ref_spatial, tensor.to_torch(out_spatial), pcc=0.999)
    assert_quality(ref_prompt, tensor.to_torch(out_prompt), pcc=0.999)


@pytest.mark.parametrize(
    "device_params",
    [{**line_params_req_exact_devices, "trace_region_size": 10000000}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("n", [4096, 7056, 6534])
@pytest.mark.parametrize("pad_value", [0.0, float("nan")], ids=["zero", "nan"])
@pytest.mark.parametrize("logical_n_kind", ["int", "tensor"])
def test_ring_joint(*, mesh_device: ttnn.MeshDevice, n: int, logical_n_kind: str, pad_value: float) -> None:
    sp_axis, tp_axis = 0, 1
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    spatial, prompt = _inputs(n)
    ref_spatial, ref_prompt = _reference(spatial, prompt)

    # Pad with NaN, to see whether padding leaks into the result.
    padded = [Attention.pad_spatial_sequence(x, sp_factor=sp_factor) for x in spatial]
    for x in padded:
        x[:, :, n:] = pad_value

    q, k, v = (tensor.from_torch(x, device=mesh_device, mesh_axes=[None, tp_axis, sp_axis, None]) for x in padded)
    add_q, add_k, add_v = (tensor.from_torch(x, device=mesh_device, mesh_axes=[None, tp_axis, None, None]) for x in prompt)

    if logical_n_kind == "int":
        logical_n = n
    else:
        logical_n = tensor.from_torch(
            torch.tensor([n]).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
        )

    grid = mesh_device.compute_with_storage_grid_size()
    out_spatial, out_prompt, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        q,
        k,
        v,
        add_q,
        add_k,
        add_v,
        persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(k.shape, 2, sp_axis),
        persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(v.shape, 2, sp_axis),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=_program_config(mesh_device),
        compute_kernel_config=_COMPUTE_CONFIG,
        dim=2,
        multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
        num_links=1,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        subdevice_id=ccl_manager.ccl_sub_device_id,
        ccl_core_grid_offset=(0, grid.y - 1),
    )

    tt_spatial = tensor.to_torch(out_spatial, mesh_axes=[None, tp_axis, sp_axis, None])
    tt_prompt = tensor.to_torch(out_prompt, mesh_axes=[None, tp_axis, None, None])
    assert_quality(ref_spatial, tt_spatial[:, :, :n], pcc=0.999)
    assert_quality(ref_prompt, tt_prompt, pcc=0.999)


def _time(device: ttnn.MeshDevice, fn: Callable[[], Any], iterations: int = 20) -> float:
    """Returns the time of one traced execution of ``fn``."""
    fn()
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = (time.perf_counter() - start) / iterations
    ttnn.release_trace(device, trace_id)
    return elapsed


@pytest.mark.parametrize(
    "device_params",
    [{**line_params, "trace_region_size": 10000000}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], ids=["1x2", "1x4"], indirect=True)
@pytest.mark.parametrize("n", [4096, 7056, 6534])
def test_ring_without_sp(*, mesh_device: ttnn.MeshDevice, n: int) -> None:
    sp_axis, tp_axis = 0, 1
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    program_config = _program_config(mesh_device)
    grid = mesh_device.compute_with_storage_grid_size()

    spatial, prompt = _inputs(n)
    ref_spatial, ref_prompt = _reference(spatial, prompt)

    q, k, v = (tensor.from_torch(x, device=mesh_device, mesh_axes=[None, tp_axis, None, None]) for x in spatial)
    # Ring attention needs whole tiles; the padding is masked by the logical length.
    ring_q, ring_k, ring_v = (
        tensor.from_torch(
            torch.nn.functional.pad(x, (0, 0, 0, -n % 512)), device=mesh_device, mesh_axes=[None, tp_axis, None, None]
        )
        for x in spatial
    )
    approx_program_config = _program_config(mesh_device, exp_approx_mode=True)
    add_q, add_k, add_v = (tensor.from_torch(x, device=mesh_device, mesh_axes=[None, tp_axis, None, None]) for x in prompt)

    def joint() -> tuple[ttnn.Tensor, ...]:
        return ttnn.transformer.joint_scaled_dot_product_attention(
            q,
            k,
            v,
            add_q,
            add_k,
            add_v,
            joint_strategy="rear",
            program_config=program_config,
            compute_kernel_config=_COMPUTE_CONFIG,
        )

    def joint_approx() -> tuple[ttnn.Tensor, ...]:
        return ttnn.transformer.joint_scaled_dot_product_attention(
            q,
            k,
            v,
            add_q,
            add_k,
            add_v,
            joint_strategy="rear",
            program_config=approx_program_config,
            compute_kernel_config=_COMPUTE_CONFIG,
        )

    def ring() -> tuple[ttnn.Tensor, ...]:
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            ring_q,
            ring_k,
            ring_v,
            add_q,
            add_k,
            add_v,
            persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(ring_k.shape, 2, sp_axis),
            persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(ring_v.shape, 2, sp_axis),
            joint_strategy="rear",
            logical_n=n,
            program_config=program_config,
            compute_kernel_config=_COMPUTE_CONFIG,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
            num_links=1,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            subdevice_id=ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(0, grid.y - 1),
        )

    for name, fn in (("joint", joint), ("joint_approx", joint_approx), ("ring", ring)):
        out_spatial, out_prompt = fn()[:2]
        out_spatial = tensor.to_torch(out_spatial, mesh_axes=[None, tp_axis, None, None])[:, :, :n]
        assert_quality(ref_spatial, out_spatial, pcc=0.999)
        assert_quality(ref_prompt, tensor.to_torch(out_prompt, mesh_axes=[None, tp_axis, None, None]), pcc=0.999)
        logger.info(f"{name} n={n}: {_time(mesh_device, fn) * 1000:.2f} ms")


@pytest.mark.parametrize(
    "device_params",
    [{**line_params, "trace_region_size": 10000000}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize(("mesh_device", "sp_axis"), [((2, 4), 0), ((1, 4), 0)], ids=["2x4sp2", "1x4sp1"], indirect=["mesh_device"])
def test_ring_traced_logical_n(*, mesh_device: ttnn.MeshDevice, sp_axis: int) -> None:
    """Captures ring attention at one logical length and replays it with others written into the tensor."""
    tp_axis = 1
    padded_length = 7168
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    grid = mesh_device.compute_with_storage_grid_size()

    def host_inputs(n: int) -> tuple[list[ttnn.Tensor], ttnn.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        spatial, prompt = _inputs(n)
        # Different data per length, so a stale replay cannot pass by accident.
        spatial = [x.roll(n, dims=-1) for x in spatial]
        reference = _reference(spatial, prompt)
        padded = [torch.nn.functional.pad(x, (0, 0, 0, padded_length - n)) for x in spatial]
        hosts = [
            tensor.from_torch(x, device=mesh_device, mesh_axes=[None, tp_axis, sp_axis, None], on_host=True)
            for x in padded
        ]
        hosts += [
            tensor.from_torch(x, device=mesh_device, mesh_axes=[None, tp_axis, None, None], on_host=True)
            for x in prompt
        ]
        logical_n = tensor.from_torch(
            torch.tensor([n]).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            on_host=True,
        )
        return hosts, logical_n, reference

    capture_n = 7056
    hosts, host_logical_n, _ = host_inputs(capture_n)
    inputs = [ttnn.to_device(h, mesh_device) for h in hosts]
    logical_n = ttnn.to_device(host_logical_n, mesh_device)
    q, k, v, add_q, add_k, add_v = inputs

    def ring() -> tuple[ttnn.Tensor, ...]:
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            add_q,
            add_k,
            add_v,
            persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(k.shape, 2, sp_axis),
            persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(v.shape, 2, sp_axis),
            joint_strategy="rear",
            logical_n=logical_n,
            program_config=_program_config(mesh_device),
            compute_kernel_config=_COMPUTE_CONFIG,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
            num_links=1,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            subdevice_id=ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(0, grid.y - 1),
        )

    ring()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_spatial, out_prompt, _ = ring()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    for n in [6534, 4096, 7056, 6534]:
        hosts, host_logical_n, (ref_spatial, ref_prompt) = host_inputs(n)
        for h, d in zip(hosts, inputs, strict=True):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.copy_host_to_device_tensor(host_logical_n, logical_n)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

        tt_spatial = tensor.to_torch(out_spatial, mesh_axes=[None, tp_axis, sp_axis, None])[:, :, :n]
        tt_prompt = tensor.to_torch(out_prompt, mesh_axes=[None, tp_axis, None, None])
        logger.info(f"replay with n={n} (captured with {capture_n})")
        assert_quality(ref_spatial, tt_spatial, pcc=0.999)
        assert_quality(ref_prompt, tt_prompt, pcc=0.999)

    # Negative control: new inputs without updating the logical length must not match.
    hosts, _, (ref_spatial, _) = host_inputs(4096)
    for h, d in zip(hosts, inputs, strict=True):
        ttnn.copy_host_to_device_tensor(h, d)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    tt_spatial = tensor.to_torch(out_spatial, mesh_axes=[None, tp_axis, sp_axis, None])[:, :, :4096]
    logger.info("control: n=4096 with logical_n left at 6534")
    with pytest.raises(Exception, match="PCC|RMSE"):  # allow-pytest.raises: scratch control
        assert_quality(ref_spatial, tt_spatial, pcc=0.999)

    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize(
    "device_params",
    [{**line_params_req_exact_devices, "trace_region_size": 10000000}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
def test_transformer_padding_finite(*, mesh_device: ttnn.MeshDevice) -> None:
    """Runs the transformer on a padded spatial sequence and checks every output row is finite."""
    from models.tt_dit.models.transformers.transformer_qwenimage import QwenImageCheckpoint
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor

    sp_axis, tp_axis = 0, 1
    latents_height, latents_width = 132, 198  # 1584 x 1056
    prompt_length = 128
    padded_length = 7168

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=4, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=2, mesh_axis=sp_axis),
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    checkpoint = QwenImageCheckpoint("Qwen/Qwen-Image")
    model = checkpoint.build(ccl_manager=ccl_manager, parallel_config=parallel_config, is_fsdp=False)

    p = checkpoint.patch_size
    n = (latents_height // p) * (latents_width // p)
    assert n == 6534
    pad = padded_length - n

    torch.manual_seed(0)
    spatial = torch.randn(1, n, 64)
    prompt = torch.randn(1, prompt_length, 3584)
    spatial_freqs, prompt_freqs = checkpoint.pos_embed(
        video_fhw=(1, latents_height // p, latents_width // p), device="cpu", max_txt_seq_len=prompt_length
    )

    def rope(freqs: torch.Tensor, *, mesh_axes: list[int | None] | None, pad: int = 0) -> tuple[ttnn.Tensor, ...]:
        return tuple(
            tensor.from_torch(
                torch.nn.functional.pad(x.repeat_interleave(2, dim=-1), (0, 0, 0, pad)),
                device=mesh_device,
                mesh_axes=mesh_axes,
            )
            for x in (freqs.real, freqs.imag)
        )

    for t in [1000, 500, 20]:
        output = model.forward(
            spatial=tensor.from_torch(
                torch.nn.functional.pad(spatial, (0, 0, 0, pad)), device=mesh_device, mesh_axes=[None, sp_axis, None]
            ),
            prompt=tensor.from_torch(prompt, device=mesh_device),
            timestep=tensor.from_torch(torch.full([1, 1], float(t)), dtype=ttnn.float32, device=mesh_device),
            spatial_rope=rope(spatial_freqs, mesh_axes=[sp_axis, None], pad=pad),
            prompt_rope=rope(prompt_freqs, mesh_axes=None),
            spatial_sequence_length=n,
        )
        out = tensor.to_torch(output, mesh_axes=[None, sp_axis, None])
        assert out.shape[1] == padded_length
        valid, padding = out[:, :n], out[:, n:]
        logger.info(
            f"t={t}: valid finite {valid.isfinite().all().item()} max |x| {valid.abs().max():.3g}, "
            f"padding finite {padding.isfinite().all().item()} max |x| {padding.abs().max():.3g}"
        )
        assert out.isfinite().all()


@pytest.mark.parametrize(
    "device_params",
    [{**line_params, "trace_region_size": 10000000}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=True)
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("n", [4096, 7056, 6534])
def test_ring_without_sp_submesh(*, mesh_device: ttnn.MeshDevice, tp: int, n: int) -> None:
    """Like `test_ring_without_sp`, on a 1 x tp submesh of the full mesh, so that fabric comes up whole."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, tp))
    test_ring_without_sp(mesh_device=submesh, n=n)
