# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pure TT-metal sampling tests inspired by vLLM sampling tests.

It should keep the high-level organization from vLLM request-level tests,
but validate behavior directly at token-ID level using synthetic logits and
on-device sampling primitives ('TTSampling' + 'SamplingGenerator').
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.sampling.tt_sampling import TTSampling

# TEST NOTES:
# - 'mesh_device' and 'device_params' come from repo-root 'conftest.py'.
# - 'device_params' defaults to '{}' (no explicit fabric config).
# - 'models/common/tests/conftest.py' provides additional common-test fixtures
#   (e.g. 'ttnn_mesh_device'), but this file uses 'mesh_device'.
# - For all multi-device paths in this file, we pass an explicit ring fabric
#   config so tests do not rely on implicit environment defaults.
# - Token ID bands are intentionally disjoint across suites for clearer failure triage.


# --- Constants & helpers ---

BATCH_SIZE = 32
MAX_TOP_K = 32
VOCAB_SIZE = 32000
FAST_NUM_TRIES = 6
FAST_NUM_STEPS = 5
MULTI_DEVICE_MESHES = [1, (4, 8)]
RING_FABRIC_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}]

# Lane positions used to sweep the "odd lane out" tests below. The sampling
# writer kernel reads each user's candidates out of a 32x32 tile where users
# 0-15 live in tile faces 0/1 and users 16-31 in faces 2/3, so cover both sides
# of that boundary as well as the first and last lane.
ODD_LANE_INDICES = [0, 15, 16, BATCH_SIZE - 1]
# The face-boundary lanes exercise per-core writer arithmetic that does not
# depend on the mesh, so multi-device runs only sweep the endpoints.
ODD_LANE_INDICES_MULTI_DEVICE = [0, BATCH_SIZE - 1]
BAND_TOKENS_PER_USER = 8


# Lightweight args container expected by TTSampling/SamplingGenerator in tests.
@dataclass
class _SamplingArgs:
    vocab_size: int
    padded_vocab_size: int
    max_batch_size: int
    max_top_k: int
    cluster_shape: tuple[int, int]
    sampling_all_gather_axis: int
    sampling_dp: int
    sub_core_grids: ttnn.CoreRangeSet | None
    sub_core_grid_topk: ttnn.CoreRangeSet | None
    start_core: ttnn.CoreCoord
    model_config: dict


def compute_per_device_vocab(vocab_size: int, num_tp: int) -> int:
    per_device = (((vocab_size + num_tp - 1) // num_tp + 31) // 32) * 32
    return 1 << (per_device - 1).bit_length()


def per_lane_params(temperature, top_k, top_p, *, size: int = BATCH_SIZE) -> SamplingParams:
    """Full-length per-lane sampling params.

    A **scalar** temperature makes format_sampling_params treat only lane 0 as active: every
    other lane takes the temperature=0.0 default and comes back GREEDY, not sampled (see
    TestFormatSamplingParamsLanes::test_all_scalar_input_is_lane_zero_plus_greedy_padding).
    That is correct, documented behaviour -- but it means any assertion about more than one
    lane sampling has to pass full-length lists, or 31 of the 32 lanes just return the argmax
    and the assertion is either vacuous or wrong. Use this whenever the whole batch is
    expected to sample; pass a scalar directly only when lane-0-only is the point. (#38316)
    """
    return SamplingParams(
        temperature=[temperature] * size,
        top_k=[top_k] * size,
        top_p=[top_p] * size,
    )


def broadcast(value, *, size: int = BATCH_SIZE):
    if isinstance(value, list):
        assert len(value) == size, f"Expected list of length {size}, got {len(value)}"
        return list(value)
    return [value] * size


def safe_sync(mesh_device):
    try:
        ttnn.synchronize_device(mesh_device)
    except Exception:
        # Cleanup best-effort only; sync failures should not mask test assertions.
        pass


def default_sub_core_grids(mesh_device) -> ttnn.CoreRangeSet:
    """The full Tensix compute grid, as the pool TTSampling carves its lane cores from.

    This must be set. ttnn.manual_seed maps seed[i] -> the core whose index in the
    enumerated core grid equals user_ids[i] (arange(max_batch_size)), and ttnn.sampling
    assigns lanes to cores over its own grid. Per-lane seeding is only correct when both
    ops see the SAME core set in the SAME order, which TTSampling guarantees by carving
    exactly max_batch_size cores out of `sub_core_grids` and passing that one set to both
    (see tt_sampling.py `_sampling_sub_core_grids`). Leaving this None makes both ops fall
    back to the unrestricted grid independently, so the mapping is no longer pinned and
    per-lane seeds land on the wrong lanes. Mirrors the construction in
    models/common/tests/test_sampling.py::_single_device_sampling_args. (#38316)
    """
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


def make_sampling_args(mesh_device, sampling_dp: int = 1) -> _SamplingArgs:
    """Build sampling args for synthetic tests on the current mesh shape."""
    cluster_shape = tuple(mesh_device.shape)
    # For 1D meshes, gather across the non-singleton axis.
    if cluster_shape[0] == 1 and cluster_shape[1] > 1:
        sampling_all_gather_axis = 1
    elif cluster_shape[1] == 1 and cluster_shape[0] > 1:
        sampling_all_gather_axis = 0
    else:
        sampling_all_gather_axis = 1 if cluster_shape[0] > 1 and cluster_shape[1] > 1 else 0
    num_tp = cluster_shape[sampling_all_gather_axis] if cluster_shape[sampling_all_gather_axis] > 0 else 1
    per_device_vocab = compute_per_device_vocab(VOCAB_SIZE, num_tp)
    padded_vocab_size = per_device_vocab * num_tp
    sub_core_grids = default_sub_core_grids(mesh_device)
    return _SamplingArgs(
        vocab_size=VOCAB_SIZE,
        padded_vocab_size=padded_vocab_size,
        max_batch_size=BATCH_SIZE,
        max_top_k=MAX_TOP_K,
        cluster_shape=cluster_shape,
        sampling_all_gather_axis=sampling_all_gather_axis,
        sampling_dp=sampling_dp,
        sub_core_grids=sub_core_grids,
        sub_core_grid_topk=sub_core_grids,
        start_core=ttnn.CoreCoord(0, 0),
        model_config={},
    )


def make_sharded_logits(torch_logits: torch.Tensor, mesh_device, args: _SamplingArgs):
    """Create device logits with vocab sharded along the sampling all-gather axis."""
    if mesh_device.get_num_devices() == 1:
        mesh_mapper = None
    elif args.cluster_shape[0] > 1 and args.cluster_shape[1] > 1:
        dims = (None, 3) if args.sampling_all_gather_axis == 1 else (3, None)
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=args.cluster_shape)
    else:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
    return ttnn.from_torch(
        torch_logits,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def infer_effective_batch_size(
    torch_logits: torch.Tensor,
    batch_size: int | None,
    *,
    max_batch_size: int = BATCH_SIZE,
) -> int:
    if torch_logits.ndim != 4:
        raise ValueError(
            f"Expected torch_logits with rank 4 [1, 1, batch, vocab], got shape {tuple(torch_logits.shape)}"
        )
    inferred_batch_size = int(torch_logits.shape[2])
    effective_batch_size = inferred_batch_size if batch_size is None else int(batch_size)
    if effective_batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {effective_batch_size}")
    if effective_batch_size > inferred_batch_size:
        raise ValueError(
            f"batch_size ({effective_batch_size}) cannot exceed logits batch dimension ({inferred_batch_size})"
        )
    if effective_batch_size > max_batch_size:
        raise ValueError(f"batch_size ({effective_batch_size}) cannot exceed max test batch size ({max_batch_size})")
    return effective_batch_size


def pad_logits_to_max_batch(torch_logits: torch.Tensor, *, max_batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Pad active-batch logits to max batch expected by TT sampling kernels."""
    current_batch = int(torch_logits.shape[2])
    if current_batch == max_batch_size:
        return torch_logits

    padded = torch.empty(
        (torch_logits.shape[0], torch_logits.shape[1], max_batch_size, torch_logits.shape[3]),
        dtype=torch_logits.dtype,
        device=torch_logits.device,
    )
    padded[:, :, :current_batch, :] = torch_logits
    # Inactive lanes are filled with a valid copy to avoid all -inf rows.
    padded[:, :, current_batch:, :] = torch_logits[:, :, :1, :]
    return padded


def validate_token_id(token: int, vocab_size: int, *, field: str):
    if not (0 <= token < vocab_size):
        raise ValueError(f"{field} token id {token} out of range [0, {vocab_size - 1}]")


def extract_tokens(tt_out_tok, batch_size: int = BATCH_SIZE, device_idx: int = 0) -> list[int]:
    """Extract token IDs from one mesh device (default device 0).

    Using a single device view matches existing TT sampling test conventions and
    keeps checks lightweight; pass device_idx to debug alternate device views.
    """
    device_tensors = ttnn.get_device_tensors(tt_out_tok)
    if not (0 <= device_idx < len(device_tensors)):
        raise ValueError(f"device_idx {device_idx} out of range for {len(device_tensors)} device tensors")
    out_torch = ttnn.to_torch(device_tensors[device_idx]).reshape(-1).to(torch.int64)
    return out_torch[:batch_size].tolist()


def extract_tokens_all_devices(tt_out_tok, batch_size: int = BATCH_SIZE) -> list[list[int]]:
    """Extract token IDs for all device views from a mesh output tensor."""
    device_tensors = ttnn.get_device_tensors(tt_out_tok)
    return [extract_tokens(tt_out_tok, batch_size=batch_size, device_idx=i) for i in range(len(device_tensors))]


def representative_device_indices(mesh_device) -> list[int]:
    """Choose a small set of device indices for cross-device consistency checks."""
    num_devices = mesh_device.get_num_devices()
    if num_devices <= 1:
        return [0]
    shape = tuple(mesh_device.shape)
    if len(shape) == 2 and shape[0] > 1 and shape[1] > 1:
        cols = shape[1]
        return [row * cols for row in range(shape[0])]
    return [0, num_devices - 1]


def build_hot_logits(
    args: _SamplingArgs,
    *,
    batch_size: int = BATCH_SIZE,
    hot_tokens: list[int] | None = None,
    per_user_hot_tokens: list[list[int]] | None = None,
    base_logit: float = -10.0,
    top_logit: float = 10.0,
    step: float = 0.25,
) -> torch.Tensor:
    """Create logits with a small hot token set that dominates sampling."""
    if hot_tokens is None and per_user_hot_tokens is None:
        raise ValueError("Either hot_tokens or per_user_hot_tokens must be provided")

    if hot_tokens is not None:
        for tok in hot_tokens:
            validate_token_id(tok, args.vocab_size, field="hot_tokens")

    if per_user_hot_tokens is None:
        per_user_hot_tokens = [list(hot_tokens)] * batch_size

    assert len(per_user_hot_tokens) == batch_size

    for user_idx, user_hot in enumerate(per_user_hot_tokens):
        for tok in user_hot:
            validate_token_id(tok, args.vocab_size, field=f"per_user_hot_tokens[{user_idx}]")

    logits = torch.full((1, 1, batch_size, args.padded_vocab_size), base_logit, dtype=torch.float32)

    for user_idx, user_hot in enumerate(per_user_hot_tokens):
        for rank, tok in enumerate(user_hot):
            logits[0, 0, user_idx, tok] = top_logit - rank * step

    logits[:, :, :, args.vocab_size :] = -float("inf")
    return logits


def build_disjoint_band_logits(
    args: _SamplingArgs,
    *,
    base_token: int,
    tokens_per_user: int = BAND_TOKENS_PER_USER,
    batch_size: int = BATCH_SIZE,
    **kwargs,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Give every user its own contiguous, non-overlapping hot-token band.

    Disjoint bands make cross-user leakage directly observable: a token returned
    for user i can only have come from user i's own logits row, so a single
    equality check localises the failure to one lane.

    Returns (logits, bands) where bands[i][0] is user i's max-logit token.
    """
    bands = [
        [base_token + user_idx * tokens_per_user + rank for rank in range(tokens_per_user)]
        for user_idx in range(batch_size)
    ]
    logits = build_hot_logits(args, per_user_hot_tokens=bands, batch_size=batch_size, **kwargs)
    return logits, bands


def build_penalty_logits(
    args: _SamplingArgs,
    *,
    target_token: int,
    batch_size: int = BATCH_SIZE,
    base_logit: float = 4.0,
    target_logit: float = 5.0,
) -> torch.Tensor:
    """Create logits where one token is the greedy target before penalties."""
    validate_token_id(target_token, args.vocab_size, field="target_token")
    logits = torch.full((1, 1, batch_size, args.padded_vocab_size), base_logit, dtype=torch.float32)
    logits[:, :, :, target_token] = target_logit
    logits[:, :, :, args.vocab_size :] = -float("inf")
    return logits


def run_ttsampling_once(
    mesh_device,
    args: _SamplingArgs,
    torch_logits: torch.Tensor,
    *,
    top_k,
    top_p,
    temperature,
    batch_size: int | None = None,
) -> list[int]:
    """Run one direct TTSampling forward and return host token IDs.

    NOTE: this bypasses 'format_sampling_params', so 'top_k'/'top_p'/'temperature'
    are the raw *device-side* values. In particular 'temperature' is the inverse
    temperature (1/T) that the kernel multiplies the logits by — it is not the
    user-facing temperature, and 0.0 is not "greedy" here. For greedy, pass
    'top_k=1' with 'temperature=1.0'.
    """
    assert all(
        t > 0 for t in broadcast(temperature)
    ), f"temperature is the device-side inverse temperature (1/T) and must be > 0, got {temperature}"
    effective_batch_size = infer_effective_batch_size(torch_logits, batch_size, max_batch_size=BATCH_SIZE)
    padded_logits = pad_logits_to_max_batch(torch_logits, max_batch_size=BATCH_SIZE)

    tt_sampling = None
    tt_input = None
    tt_logits = None
    tt_tokens = None
    tt_log_probs = None

    try:
        tt_sampling = TTSampling(
            mesh_device=mesh_device,
            tt_ccl=None,
            args=args,
            k=torch.tensor(broadcast(top_k), dtype=torch.int64),
            p=torch.tensor(broadcast(top_p), dtype=torch.float32),
            temp=torch.tensor(broadcast(temperature), dtype=torch.float32),
        )
        tt_logits = padded_logits
        tt_input = make_sharded_logits(tt_logits, mesh_device, args)
        tt_tokens, tt_log_probs = tt_sampling(tt_input)
        return extract_tokens(tt_tokens, effective_batch_size)
    finally:
        if tt_log_probs is not None:
            del tt_log_probs

        if tt_tokens is not None:
            del tt_tokens

        if tt_input is not None:
            del tt_input

        if tt_logits is not None:
            del tt_logits

        if tt_sampling is not None:
            del tt_sampling

        safe_sync(mesh_device)


def run_sampling_generator(
    mesh_device,
    args: _SamplingArgs,
    torch_logits: torch.Tensor,
    sampling_params: SamplingParams,
    *,
    num_steps: int = 1,
    advance_seeds: bool = True,
    seed_values: list[int] | None = None,
    write_seed_values_to_device: bool = False,
    batch_size: int | None = None,
    device_idx: int = 0,
    state_setup=None,
    enable_trace: bool = False,
) -> list[list[int]]:
    """Run SamplingGenerator for num_steps and return per-step token lists."""
    effective_batch_size = infer_effective_batch_size(torch_logits, batch_size, max_batch_size=BATCH_SIZE)
    padded_logits = pad_logits_to_max_batch(torch_logits, max_batch_size=BATCH_SIZE)
    sg = None
    tt_input = None
    tt_tokens = None
    tt_log_probs = None
    outputs = []

    try:
        sg = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None)
        formatted = format_sampling_params(sampling_params, BATCH_SIZE)
        sg.reset_sampling_params(formatted)

        if seed_values is not None:
            if len(seed_values) > BATCH_SIZE:
                raise ValueError(f"seed_values length ({len(seed_values)}) cannot exceed BATCH_SIZE ({BATCH_SIZE})")
            if len(seed_values) > effective_batch_size and not write_seed_values_to_device:
                raise ValueError(
                    f"seed_values length ({len(seed_values)}) cannot exceed active batch size ({effective_batch_size})"
                )
            user_ids = list(range(len(seed_values)))
            sg.seed_manager.reset_seed(seed_values, user_ids)
        else:
            # Mirror production's apply_prefill_state: reset_sampling_params does
            # not touch the SeedManager, so an unseeded request must still call
            # reset_seed (seeds=None) to move it out of the fresh _reseted=False
            # state. Without this the SeedManager stays in the steady "SKIP" state
            # and get_new_values() never pushes entropy seeds, making unseeded
            # sampling deterministic.
            sg.seed_manager.reset_seed(None, list(range(BATCH_SIZE)))

        if state_setup is not None:
            state_setup(sg)

        if enable_trace and sg._penalties_active:
            raise ValueError(
                "enable_trace=True is incompatible with penalties in this harness: TTPenalties.apply() "
                "rewrites the traced input tensor in place, and trace replay cannot re-upload it."
            )

        if write_seed_values_to_device:
            if seed_values is None:
                raise ValueError("write_seed_values_to_device=True requires seed_values")
            sg.seed_manager.write_device_seed_values(seed_values)

        # Trace replay binds the captured input address, so `_validate_trace_inputs`
        # requires the SAME logits tensor object on every step after capture. Upload it
        # once up front for traced runs; penalties are rejected below because
        # TTPenalties.apply() would rewrite that one tensor in place.
        if enable_trace:
            tt_input = make_sharded_logits(padded_logits, mesh_device, args)

        for _ in range(num_steps):
            # SamplingGenerator keeps per-user RNG state in SeedManager.
            if advance_seeds:
                sg.seed_manager.get_new_values()

            if not enable_trace:
                # TTPenalties.apply() rewrites the logits tensor in place, so upload a
                # pristine copy every step. Reusing one device tensor would compound
                # each step's penalties onto the previous step's already-penalised
                # logits, whereas real decode gets fresh logits from the LM head.
                if tt_input is not None:
                    ttnn.deallocate(tt_input)
                tt_input = make_sharded_logits(padded_logits, mesh_device, args)

            tt_tokens, tt_log_probs = sg.sample(tt_input, enable_trace=enable_trace)
            if enable_trace:
                # ttnn.execute_trace replays with blocking=False, so the output buffer is
                # only valid once the device has caught up.
                ttnn.synchronize_device(mesh_device)
            outputs.append(extract_tokens(tt_tokens, effective_batch_size, device_idx=device_idx))

        return outputs
    finally:
        # Release the captured trace before the generator goes away. mesh_device is
        # function-scoped so the region would be reclaimed at device close anyway, but a
        # leaked trace breaks any future test that captures twice in one device session.
        if enable_trace and sg is not None:
            try:
                sg.reset_trace()
            except Exception:
                pass

        if tt_log_probs is not None:
            del tt_log_probs

        if tt_tokens is not None:
            del tt_tokens

        if tt_input is not None:
            del tt_input

        if sg is not None:
            del sg

        safe_sync(mesh_device)


def assert_tokens_in_vocab(tokens: list[int], vocab_size: int = VOCAB_SIZE):
    assert all(
        0 <= tok < vocab_size for tok in tokens
    ), f"Found out-of-range token(s) for vocab_size={vocab_size}: {tokens}"


def flatten_steps(outputs: list[list[int]]) -> list[int]:
    return [tok for step in outputs for tok in step]


# --- Test: prefill parameter behavior ---


class TestPrefillWithDifferentParams:
    @pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
    @pytest.mark.parametrize(
        "device_params",
        RING_FABRIC_DEVICE_PARAMS,
        indirect=True,
    )
    def test_prefill_temperature_varied_in_batch(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        hot_tokens = [100, 101, 102, 103, 104, 105, 106, 107]
        hot_token_set = set(hot_tokens)
        logits = build_hot_logits(args, hot_tokens=hot_tokens)
        # Every lane gets a different temperature, including greedy lanes (0.0).
        temperature = ([0.0, 0.5, 1.0, 2.0] * (BATCH_SIZE // 4))[:BATCH_SIZE]
        greedy_lanes = [i for i, t in enumerate(temperature) if t == 0.0]
        params = SamplingParams(temperature=temperature, top_k=8, top_p=1.0)
        seeds = [3100 + i for i in range(BATCH_SIZE)]
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        assert out1 == out2, "Same per-user seeds should replay exactly for stochastic prefill config"
        assert_tokens_in_vocab(out1, args.vocab_size)
        unexpected = [tok for tok in out1 if tok not in hot_token_set]
        assert not unexpected, f"Sampled tokens outside expected hot set: {unexpected}, out={out1}"
        for lane in greedy_lanes:
            assert out1[lane] == hot_tokens[0], (
                f"temperature=0.0 lane {lane} must be greedy on the max-logit token "
                f"{hot_tokens[0]}, got {out1[lane]}. out={out1}"
            )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_prefill_temperature_varied_between_batches(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[120, 121, 122, 123, 124, 125, 126, 127])
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        outputs = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=FAST_NUM_TRIES, advance_seeds=True
        )
        user0 = [step[0] for step in outputs]
        assert len(set(user0)) >= 2, f"Expected variation across runs for stochastic sampling, got {user0}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_prefill_topk_mixed_greedy_and_stochastic(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[140, 141, 142, 143, 144, 145, 146, 147])

        half = BATCH_SIZE // 2
        temperature = [0.0] * half + [1.5] * (BATCH_SIZE - half)  # First half is greedy; second half is stochastic.
        top_k = [32] * half + [8] * (BATCH_SIZE - half)
        top_p = [1.0] * BATCH_SIZE
        params = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)

        seed_values = [1000 + i for i in range(BATCH_SIZE)]
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seed_values
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seed_values
        )[0]

        assert out1 == out2, "Same per-user seeds must replay exactly across batches"
        greedy = out1[:half]
        stochastic = out1[half:]
        assert len(set(greedy)) == 1, f"Greedy half should be deterministic and identical, got {greedy}"
        assert len(set(stochastic)) >= 2, f"Stochastic half should vary across slots, got {stochastic}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_prefill_seeding(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[160, 161, 162, 163, 164, 165, 166, 167])

        thirds = BATCH_SIZE // 3
        greedy_count = BATCH_SIZE - 2 * thirds
        temperature = [0.0] * greedy_count + [1.5] * (BATCH_SIZE - greedy_count)
        top_k = [32] * greedy_count + [8] * (BATCH_SIZE - greedy_count)
        top_p = [1.0] * BATCH_SIZE
        params = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)

        seed_values = [2000 + i for i in range(BATCH_SIZE)]
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seed_values
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seed_values
        )[0]
        assert out1 == out2, "Replay with same seeds must be deterministic for every slot"
        assert len(set(out1[greedy_count:])) >= 2, "Different stochastic seeds should yield diverse tokens"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_prefill_topk_1_is_greedy(self, mesh_device):
        args = make_sampling_args(mesh_device)
        hot_tokens = [180, 181, 182, 183]
        logits = build_hot_logits(args, hot_tokens=hot_tokens)
        # top_k=1 collapses to argmax whatever the temperature scaling is, so the
        # max-logit token must come back for every inverse temperature.
        # top_p=1.0 (disabled) keeps this portable across top-p conventions.
        for inverse_temperature in (0.25, 1.0, 4.0):
            tokens = run_ttsampling_once(mesh_device, args, logits, top_k=1, top_p=1.0, temperature=inverse_temperature)
            assert all(tok == hot_tokens[0] for tok in tokens), (
                f"top_k=1 should be greedy at inverse temperature {inverse_temperature}, "
                f"expected all {hot_tokens[0]}, got {tokens}"
            )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_greedy_picks_max_logit(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[42, 43, 44])  # 42 has highest logit
        tokens = run_ttsampling_once(mesh_device, args, logits, top_k=1, top_p=1.0, temperature=1.0)
        assert all(tok == 42 for tok in tokens), "Greedy should always pick the max logit token"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_run_ttsampling_once_respects_logits_batch_size(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, batch_size=2, hot_tokens=[60, 61, 62])
        tokens = run_ttsampling_once(mesh_device, args, logits, top_k=1, top_p=1.0, temperature=1.0)
        assert len(tokens) == 2, f"Expected 2 tokens for batch_size=2 logits, got {len(tokens)}"
        assert_tokens_in_vocab(tokens, args.vocab_size)
        assert all(tok == 60 for tok in tokens), f"Greedy should pick the max-logit token 60, got {tokens}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_run_sampling_generator_respects_logits_batch_size(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, batch_size=2, hot_tokens=[80, 81, 82])
        params = SamplingParams(temperature=[0.0, 1.0], top_k=[1, 3], top_p=[1.0, 1.0])
        outputs = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)
        assert len(outputs) == 1
        assert len(outputs[0]) == 2, f"Expected 2 tokens for batch_size=2 logits, got {len(outputs[0])}"
        assert_tokens_in_vocab(outputs[0], args.vocab_size)

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_extract_tokens_invalid_device_idx_raises(self, mesh_device, expect_error):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[84, 85, 86])
        params = SamplingParams(temperature=1.0, top_k=3, top_p=1.0)
        invalid_device_idx = mesh_device.get_num_devices()
        with expect_error(ValueError, r"device_idx .* out of range"):
            run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=1,
                advance_seeds=False,
                device_idx=invalid_device_idx,
            )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_build_hot_logits_rejects_out_of_vocab_token(self, mesh_device, expect_error):
        args = make_sampling_args(mesh_device)
        bad_token = args.vocab_size
        with expect_error(ValueError, r"out of range"):
            build_hot_logits(args, hot_tokens=[bad_token])

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_build_penalty_logits_rejects_out_of_vocab_target(self, mesh_device, expect_error):
        args = make_sampling_args(mesh_device)
        bad_token = args.vocab_size
        with expect_error(ValueError, r"out of range"):
            build_penalty_logits(args, target_token=bad_token)

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_top_p_restricts_candidate_set(self, mesh_device):
        args = make_sampling_args(mesh_device)
        hot_tokens = [1900, 1901, 1902]
        logits = build_hot_logits(
            args,
            hot_tokens=hot_tokens,
            base_logit=-12.0,
            top_logit=2.0,
            step=0.2,
        )
        seeds = [31000 + i for i in range(BATCH_SIZE)]

        # Per-lane lists so all 32 lanes sample. With a scalar temperature only lane 0 would,
        # making "1902 in full_set" a 6-draw coin flip (~15% failure) instead of 192 draws.
        params_low = per_lane_params(1.0, 3, 0.6)
        params_full = per_lane_params(1.0, 3, 1.0)

        out_low = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params_low,
            num_steps=FAST_NUM_TRIES,
            advance_seeds=True,
            seed_values=seeds,
        )
        out_full = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params_full,
            num_steps=FAST_NUM_TRIES,
            advance_seeds=True,
            seed_values=seeds,
        )

        for step_tokens in out_low + out_full:
            assert_tokens_in_vocab(step_tokens, args.vocab_size)

        low_flat = flatten_steps(out_low)
        full_flat = flatten_steps(out_full)
        low_set = set(low_flat)
        full_set = set(full_flat)

        allowed_low = {1900, 1901}
        allowed_full = {1900, 1901, 1902}
        assert low_set.issubset(allowed_low), (
            f"top_p=0.6 sampled outside allowed set {sorted(allowed_low)}. "
            f"low_set={sorted(low_set)}, low_hist={Counter(low_flat).most_common(6)}"
        )
        assert full_set.issubset(allowed_full), (
            f"top_p=1.0 sampled outside expected hot set {sorted(allowed_full)}. "
            f"full_set={sorted(full_set)}, full_hist={Counter(full_flat).most_common(6)}"
        )
        assert 1902 in full_set, (
            "top_p=1.0 did not sample token 1902 at least once. "
            f"low_set={sorted(low_set)}, full_set={sorted(full_set)}, "
            f"low_hist={Counter(low_flat).most_common(6)}, full_hist={Counter(full_flat).most_common(6)}"
        )


# --- Test: per-request penalties ---


@pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    RING_FABRIC_DEVICE_PARAMS,
    indirect=True,
)
class TestRepetitionPenaltyPerRequest:
    def test_different_repetition_penalties(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 500
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = ([1.0, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0] * 4)[:BATCH_SIZE]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(
            mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token)
        )[0]
        low_penalty = [
            tok for i, tok in enumerate(tokens) if penalties[i] <= 1.0
        ]  # Keep boundary-sensitive values (e.g. 1.2) out of strict assertions.
        high_penalty = [tok for i, tok in enumerate(tokens) if penalties[i] >= 1.5]
        assert all(tok == target_token for tok in low_penalty), "Low repetition penalty should keep target token"
        assert any(tok != target_token for tok in high_penalty), "High repetition penalties should alter output"

    def test_repetition_penalty_vs_no_penalty(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 520
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [1.0 if i % 2 == 0 else 2.5 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(
            mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token)
        )[0]
        no_penalty = [tokens[i] for i in range(0, BATCH_SIZE, 2)]
        with_penalty = [tokens[i] for i in range(1, BATCH_SIZE, 2)]
        assert all(tok == target_token for tok in no_penalty), "No-penalty lanes should keep target"
        assert all(tok != target_token for tok in with_penalty), "Penalty lanes should change token"
        assert no_penalty[0] != with_penalty[0], "Penalty and no-penalty outputs should differ"

    def test_repetition_penalty_persists_across_steps(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 540
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [1.0 if i % 2 == 0 else 2.5 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        outputs = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=3,
            advance_seeds=False,
        )

        for step_tokens in outputs:
            assert_tokens_in_vocab(step_tokens, args.vocab_size)

        even_idxs = range(0, BATCH_SIZE, 2)
        odd_idxs = range(1, BATCH_SIZE, 2)
        assert all(outputs[0][i] == target_token for i in odd_idxs), "Odd lanes should start at target token"
        assert all(outputs[0][i] == target_token for i in even_idxs), "Even lanes should start at target token"
        assert all(outputs[1][i] != target_token for i in odd_idxs), "Odd lanes should leave target at step 1"
        assert all(outputs[2][i] != target_token for i in odd_idxs), "Odd lanes should stay off target at step 2"
        assert all(outputs[1][i] == target_token for i in even_idxs), "Even lanes should keep target at step 1"
        assert all(outputs[2][i] == target_token for i in even_idxs), "Even lanes should keep target at step 2"

    def _state_setup(self, sg, target_token: int):
        seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
        sg.reset_prompt_tokens(seen)
        sg.reset_output_state(tokens=seen)

    def _get_sampling_params(self, penalties):
        return SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            repetition_penalty=penalties,
        )


@pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    RING_FABRIC_DEVICE_PARAMS,
    indirect=True,
)
class TestPresencePenaltyPerRequest:
    def test_different_presence_penalties(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 700
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = ([0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, 4.0] * 4)[:BATCH_SIZE]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(
            mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token)
        )[0]
        assert any(tok == target_token for tok in tokens), "Some lanes should retain target token"
        assert any(tok != target_token for tok in tokens), "Some lanes should shift off target with higher penalties"

    def test_presence_penalty_mixed_batch(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 720
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.0 if i % 2 == 0 else 2.0 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(
            mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token)
        )[0]
        no_penalty = [tokens[i] for i in range(0, BATCH_SIZE, 2)]
        with_penalty = [tokens[i] for i in range(1, BATCH_SIZE, 2)]
        assert all(tok == target_token for tok in no_penalty), "No presence-penalty lanes should keep target"
        assert all(tok != target_token for tok in with_penalty), "Presence-penalty lanes should move off target"

    def test_presence_penalty_persists_across_steps(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 740
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.0 if i % 2 == 0 else 2.0 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        outputs = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=3,
            advance_seeds=False,
        )

        for step_tokens in outputs:
            assert_tokens_in_vocab(step_tokens, args.vocab_size)

        even_idxs = range(0, BATCH_SIZE, 2)
        odd_idxs = range(1, BATCH_SIZE, 2)
        assert all(outputs[0][i] == target_token for i in odd_idxs), "Odd lanes should start at target token"
        assert all(outputs[0][i] == target_token for i in even_idxs), "Even lanes should start at target token"
        assert all(outputs[1][i] != target_token for i in odd_idxs), "Odd lanes should leave target at step 1"
        assert all(outputs[2][i] != target_token for i in odd_idxs), "Odd lanes should stay off target at step 2"
        assert all(outputs[1][i] == target_token for i in even_idxs), "Even lanes should keep target at step 1"
        assert all(outputs[2][i] == target_token for i in even_idxs), "Even lanes should keep target at step 2"

    def _state_setup(self, sg, target_token: int):
        seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
        sg.reset_prompt_tokens(seen)
        sg.reset_output_state(tokens=seen)

    def _get_sampling_params(self, penalties):
        return SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            presence_penalty=penalties,
        )


@pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    RING_FABRIC_DEVICE_PARAMS,
    indirect=True,
)
class TestFrequencyPenaltyPerRequest:
    def test_different_frequency_penalties(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 900
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = ([0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, 4.0] * 4)[:BATCH_SIZE]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(
            mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token)
        )[0]
        assert any(tok == target_token for tok in tokens), "Some lanes should retain target token"
        assert any(tok != target_token for tok in tokens), "Some lanes should shift off target with higher penalties"

    def test_frequency_penalty_mixed_batch(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 920
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.0 if i % 2 == 0 else 2.0 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(
            mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token)
        )[0]
        no_penalty = [tokens[i] for i in range(0, BATCH_SIZE, 2)]
        with_penalty = [tokens[i] for i in range(1, BATCH_SIZE, 2)]
        assert all(tok == target_token for tok in no_penalty), "No frequency-penalty lanes should keep target"
        assert all(tok != target_token for tok in with_penalty), "Frequency-penalty lanes should move off target"

    def test_frequency_penalty_accumulates_across_steps(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 940
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.1 if i % 2 == 0 else 0.6 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        outputs = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=4,
            advance_seeds=False,
        )

        for step_tokens in outputs:
            assert_tokens_in_vocab(step_tokens, args.vocab_size)

        even_idxs = range(0, BATCH_SIZE, 2)
        odd_idxs = range(1, BATCH_SIZE, 2)
        assert all(outputs[0][i] == target_token for i in odd_idxs), "Odd lanes should start at target token"
        assert all(outputs[1][i] == target_token for i in odd_idxs), "Odd lanes should still hit target at step 1"
        assert all(outputs[2][i] != target_token for i in odd_idxs), "Odd lanes should leave target at step 2"
        assert all(outputs[3][i] != target_token for i in odd_idxs), "Odd lanes should stay off target at step 3"
        assert all(outputs[0][i] == target_token for i in even_idxs), "Even lanes should start at target token"
        assert all(outputs[1][i] == target_token for i in even_idxs), "Even lanes should keep target at step 1"
        assert all(outputs[2][i] == target_token for i in even_idxs), "Even lanes should keep target at step 2"
        assert all(outputs[3][i] == target_token for i in even_idxs), "Even lanes should keep target at step 3"

    def _state_setup(self, sg, target_token: int):
        seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
        sg.reset_prompt_tokens(seen)
        sg.reset_output_state(tokens=seen)

    def _get_sampling_params(self, penalties):
        return SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            frequency_penalty=penalties,
        )


# --- Test: seed behavior ---
class TestSeededSamplingPerRequest:
    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_run_sampling_generator_rejects_seed_vector_longer_than_max_batch(self, mesh_device, expect_error):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[990, 991, 992])
        params = SamplingParams(temperature=1.0, top_k=3, top_p=1.0)
        too_many_seeds = list(range(BATCH_SIZE + 1))
        with expect_error(ValueError, r"cannot exceed BATCH_SIZE"):
            run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=1,
                advance_seeds=True,
                seed_values=too_many_seeds,
            )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_run_sampling_generator_rejects_seed_vector_longer_than_effective_batch(self, mesh_device, expect_error):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, batch_size=2, hot_tokens=[993, 994, 995])
        params = SamplingParams(temperature=[1.0, 1.0], top_k=[3, 3], top_p=[1.0, 1.0])
        too_many_for_active_batch = [11, 12, 13]
        with expect_error(ValueError, r"active batch size"):
            run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=1,
                advance_seeds=True,
                seed_values=too_many_for_active_batch,
            )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_seeded_replay_is_deterministic(self, mesh_device):
        """Same seed vector must replay exactly across independent generator instances.

        Split out of the per-lane seed investigation below so that these invariants --
        replay determinism, token range, and top-k containment -- keep gating CI while the
        per-lane seed question is still open. A regression to argmax-only sampling or an
        out-of-range token fails here, not silently as an xfail. (#38316)
        """
        args = make_sampling_args(mesh_device)
        hot_tokens = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
        hot_token_set = set(hot_tokens)
        logits = build_hot_logits(args, hot_tokens=hot_tokens)
        params = per_lane_params(1.5, 8, 1.0)

        seeds_a = list(range(BATCH_SIZE))
        out_a1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_a
        )[0]
        out_a2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_a
        )[0]

        assert out_a1 == out_a2, "Same seed vector should replay exactly across independent runs"

        for outputs in (out_a1, out_a2):
            assert_tokens_in_vocab(outputs, args.vocab_size)
            unexpected = [tok for tok in outputs if tok not in hot_token_set]
            assert not unexpected, (
                f"Sampled tokens outside expected hot set {sorted(hot_token_set)}: {unexpected}. " f"outputs={outputs}"
            )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_different_seeds_produce_different_outputs(self, mesh_device):
        """Distinct per-lane seeds must produce a distinct draw per lane.

        Previously xfailed as a suspected per-lane seeding defect (blamed on #50685). It was a
        bug in this test: it passed SCALAR sampling params, so format_sampling_params treated
        only lane 0 as active and lanes 1..31 came back greedy on the argmax. Run 32289587326
        with the marker removed showed it exactly -- out_a1 = [1002, 1000, 1000, ... 1000] with
        hot_tokens[0] == 1000 -- so the "distinct seeds" assertion was comparing 31 deterministic
        argmax lanes plus one stochastic lane whose single draw happened to repeat. Fixed by
        passing per-lane lists; there is no product defect and #50685 is not implicated.
        """
        args = make_sampling_args(mesh_device)
        hot_tokens = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
        logits = build_hot_logits(args, hot_tokens=hot_tokens)
        # Per-lane lists: a scalar temperature would leave lanes 1..31 greedy and this test
        # would be asserting against 31 argmax lanes. See per_lane_params.
        params = per_lane_params(1.5, 8, 1.0)

        seeds_a = list(range(BATCH_SIZE))
        out_a1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_a
        )[0]

        # 32 distinct seeds over a flat-ish top-8 distribution: the lanes must not all
        # land on the same token. This is the reported symptom -- lanes 1..31 coming out
        # identical despite distinct seeds.
        assert len(set(out_a1)) > 1, (
            f"32 distinct per-lane seeds produced one token across all lanes: {out_a1}. "
            "Per-lane seeding is not taking effect."
        )

        # Shifting every seed must move at least one lane; identical output for a different
        # seed vector means the RNG path is inert (argmax-only).
        seeds_b = [s + 12345 for s in seeds_a]
        out_b = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_b
        )[0]
        changed_indices = [i for i, (a, b) in enumerate(zip(out_a1, out_b)) if a != b]
        assert len(changed_indices) > 0, (
            "Different seed vector produced identical outputs for this stochastic config; "
            f"backend appears argmax-only (seed/RNG regression). out_a1={out_a1}, out_b={out_b}"
        )

    @pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
    @pytest.mark.parametrize(
        "device_params",
        RING_FABRIC_DEVICE_PARAMS,
        indirect=True,
    )
    def test_same_seeds_reproduce_across_batches(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107])
        params = per_lane_params(1.25, 8, 1.0)
        seeds = [500 + i for i in range(BATCH_SIZE)]
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=FAST_NUM_STEPS, advance_seeds=True, seed_values=seeds
        )
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=FAST_NUM_STEPS, advance_seeds=True, seed_values=seeds
        )
        assert out1 == out2, "Same seed replay should match exactly for all users and steps"

        for device_idx in representative_device_indices(mesh_device)[1:]:
            out_device = run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=FAST_NUM_STEPS,
                advance_seeds=True,
                seed_values=seeds,
                device_idx=device_idx,
            )
            assert out_device == out1, f"Device view mismatch for device_idx={device_idx} in seeded replay"

    @pytest.mark.parametrize("seed", [42, 123, 999, 0])
    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_specific_seed_reproducible(self, mesh_device, seed):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207])
        params = SamplingParams(temperature=1.25, top_k=8, top_p=1.0)
        seeds = [10000 + i for i in range(BATCH_SIZE)]
        seeds[0] = seed
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        assert out1[0] == out2[0], f"Seed {seed} should be reproducible for user-0"

    @pytest.mark.parametrize("seed", [1, 0])
    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_batch1_seed_reproducible(self, mesh_device, seed):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307])
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        results = []

        for _ in range(FAST_NUM_TRIES):
            seeds = [seed] + [100000 + i for i in range(1, BATCH_SIZE)]
            out = run_sampling_generator(
                mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
            )[0]
            results.append(out[0])

        assert len(set(results)) == 1, f"Single seeded slot should reproduce, got {results}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_batch1_no_seed_varied(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407])
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        results = []

        for _ in range(FAST_NUM_TRIES):
            out = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)[0]
            results.append(out[0])

        assert len(set(results)) >= 2, f"Unseeded single slot should vary across requests, got {results}"

    @pytest.mark.parametrize("seed", [1, 0])
    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_uniform_seed_deterministic(self, mesh_device, seed):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507])
        params = per_lane_params(1.0, 8, 1.0)
        seeds = [seed] * BATCH_SIZE
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        assert out1 == out2, f"Uniform seed {seed} should be deterministic across runs"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_uniform_noseed_varied(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607])
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        outputs = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=FAST_NUM_STEPS, advance_seeds=True
        )
        user0 = [step[0] for step in outputs]
        assert len(set(user0)) >= 2, f"Expected unseeded variation across steps, got {user0}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_seed_0_produces_deterministic_outputs(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707])
        params = SamplingParams(temperature=1.0, top_k=8, top_p=1.0)
        seeds = [0] * BATCH_SIZE
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        assert out1 == out2, "Seed 0 must be deterministic across requests"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_negative_seed_does_not_crash(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807])
        params = SamplingParams(temperature=1.0, top_k=8, top_p=1.0)
        seeds = [-1] + [2000 + i for i in range(1, BATCH_SIZE)]
        out = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=1,
            advance_seeds=False,
            seed_values=seeds,
            write_seed_values_to_device=True,
        )[0]
        assert_tokens_in_vocab(out, args.vocab_size)


# --- Test: batch isolation ---


@pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    RING_FABRIC_DEVICE_PARAMS,
    indirect=True,
)
class TestBatchIsolation:
    def test_mixed_params_batch(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007])

        temperature_template = [0.0, 1.5, 0.5, 0.5, 1.0, 0.01, 2.0, 0.7]
        top_k_template = [32, 8, 8, 8, 1, 32, 5, 8]
        top_p_template = [1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 0.9]
        repetition_template = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.5]
        presence_template = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]
        frequency_template = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]

        temperature = (temperature_template * 4)[:BATCH_SIZE]
        top_k = (top_k_template * 4)[:BATCH_SIZE]
        top_p = (top_p_template * 4)[:BATCH_SIZE]
        repetition = (repetition_template * 4)[:BATCH_SIZE]
        presence = (presence_template * 4)[:BATCH_SIZE]
        frequency = (frequency_template * 4)[:BATCH_SIZE]
        seeds = [9000 + i for i in range(BATCH_SIZE)]

        params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition,
            presence_penalty=presence,
            frequency_penalty=frequency,
        )

        out1 = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=1,
            advance_seeds=True,
            seed_values=seeds,
            state_setup=self._state_setup,
        )[0]
        out2 = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=1,
            advance_seeds=True,
            seed_values=seeds,
            state_setup=self._state_setup,
        )[0]
        assert out1 == out2, "Deterministic lanes (and seeded lanes) should replay identically across runs"
        assert len(set(out1)) >= 2, "Mixed batch should not collapse to a single token"

    def test_outputs_not_mixed_different_prompts(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        tokens_per_user = 4
        per_user_hot = []
        expected_sets = []
        token_cursor = 2200
        for _ in range(BATCH_SIZE):
            hot = [token_cursor + i for i in range(tokens_per_user)]
            per_user_hot.append(hot)
            expected_sets.append(set(hot))
            token_cursor += tokens_per_user

        logits = build_hot_logits(args, per_user_hot_tokens=per_user_hot)
        params = SamplingParams(temperature=1.0, top_k=tokens_per_user, top_p=1.0)
        seeds = [10000 + i for i in range(BATCH_SIZE)]
        tokens = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        for i, tok in enumerate(tokens):
            assert (
                tok in expected_sets[i]
            ), f"User {i} token leaked across users: tok={tok}, expected={expected_sets[i]}"

        for device_idx in representative_device_indices(mesh_device)[1:]:
            device_tokens = run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=1,
                advance_seeds=True,
                seed_values=seeds,
                device_idx=device_idx,
            )[0]
            assert device_tokens == tokens, f"Device view mismatch for device_idx={device_idx} in batch isolation test"

    def test_same_prompt_users_get_identical_logits(self, mesh_device, device_params):
        """Identical prompts must agree under greedy, and vary under distinct seeds.

        Split out of the uniform-seed sub-case below (#38316): these two invariants have
        nothing to do with per-lane seeding and must keep gating CI.
        """
        args = make_sampling_args(mesh_device)
        # All users share the exact same hot-token distribution.
        hot_tokens = [2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507]
        logits = build_hot_logits(args, hot_tokens=hot_tokens)

        # --- Greedy: every user should pick the top-logit token. ---
        greedy_params = SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
        )
        greedy_tokens = run_sampling_generator(
            mesh_device, args, logits, greedy_params, num_steps=1, advance_seeds=False
        )[0]
        assert (
            len(set(greedy_tokens)) == 1
        ), f"All users with the same prompt under greedy should pick the same token, got {greedy_tokens}"
        assert (
            greedy_tokens[0] == hot_tokens[0]
        ), f"Greedy should pick the highest-logit token {hot_tokens[0]}, got {greedy_tokens[0]}"

        # --- Stochastic with different seeds: should see variation. ---
        # Use full-length lists (like the greedy sub-case above) so temperature applies
        # uniformly to every lane. A scalar temperature would only configure lane 0 and
        # leave lanes 1..31 on the greedy default.
        stochastic_params = SamplingParams(
            temperature=[1.5] * BATCH_SIZE,
            top_k=[8] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
        )
        diverse_seeds = [5000 + i for i in range(BATCH_SIZE)]
        out_diverse = run_sampling_generator(
            mesh_device,
            args,
            logits,
            stochastic_params,
            num_steps=FAST_NUM_TRIES,
            advance_seeds=True,
            seed_values=diverse_seeds,
        )
        all_tokens = flatten_steps(out_diverse)
        assert (
            len(set(all_tokens)) >= 2
        ), f"Different seeds on the same prompt should produce variation, got {set(all_tokens)}"
        assert_tokens_in_vocab(all_tokens, args.vocab_size)
        unexpected = [tok for tok in all_tokens if tok not in set(hot_tokens)]
        assert not unexpected, f"Sampled tokens outside expected hot set: {unexpected}"

    def test_uniform_seed_diverges_across_lanes(self, mesh_device, device_params):
        """One request seed shared by every lane must still give each lane its own draw.

        This asserts the SEED SALT, not agreement. SeedManager._set_slot_seed assigns
        seed_salts[slot] = _next_free_salt(slot, seed): the first slot holding a given seed
        gets salt 0 and each later duplicate gets the next free value, and the salt is mixed
        into _hash_request_seed_to_device_seed. Without it, n>1 completions of one prompt at
        a fixed seed came out byte-identical (#53077). So a uniform seed vector is exactly the
        case the salt exists to separate, and identical tokens across lanes would mean the
        salt is not being applied.

        A request whose seed is unique among live slots always lands on salt 0, so
        single-seeded reproducibility (test_specific_seed_reproducible,
        test_seeded_replay_is_deterministic) is unaffected.

        Deliberately paired with
        test_sampling.py::test_ttsampling_duplicate_request_seeds_sample_diverse_tokens, which
        asserts the same invariant one level down on raw TTSampling + SeedManager. This one
        goes through SamplingGenerator and format_sampling_params, so it also covers the
        wiring between them. Change one, check the other.
        """
        args = make_sampling_args(mesh_device)
        hot_tokens = [2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507]
        hot_token_set = set(hot_tokens)
        logits = build_hot_logits(args, hot_tokens=hot_tokens)
        stochastic_params = SamplingParams(
            temperature=[1.5] * BATCH_SIZE,
            top_k=[8] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
        )
        uniform_seed = [7777] * BATCH_SIZE
        out_uniform = run_sampling_generator(
            mesh_device,
            args,
            logits,
            stochastic_params,
            num_steps=1,
            advance_seeds=True,
            seed_values=uniform_seed,
        )[0]

        assert len(set(out_uniform)) > 1, (
            "Every lane sharing seed 7777 produced the same token, so the per-slot seed salt "
            f"is not reaching the device: {out_uniform}. This is the #53077 failure mode."
        )
        assert_tokens_in_vocab(out_uniform, args.vocab_size)
        unexpected = [tok for tok in out_uniform if tok not in hot_token_set]
        assert not unexpected, f"Sampled tokens outside expected hot set: {unexpected}"

        # Replay must still be exact: the salt is derived from the live slot layout, which is
        # identical for a fresh generator given the same seed vector.
        out_again = run_sampling_generator(
            mesh_device,
            args,
            logits,
            stochastic_params,
            num_steps=1,
            advance_seeds=True,
            seed_values=uniform_seed,
        )[0]
        assert out_uniform == out_again, (
            "Salted per-slot seeds must still replay exactly for the same seed vector. "
            f"first={out_uniform}, second={out_again}"
        )

    def _state_setup(self, sg):
        seen = torch.full((BATCH_SIZE, 1), 2000, dtype=torch.int64)
        sg.reset_prompt_tokens(seen)
        sg.reset_output_state(tokens=seen)


# --- Test: batch size variations ---


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
class TestBatchSizeVariations:
    def _make_active_user_logits(self, args: _SamplingArgs, active_users: int):
        per_user_hot = []
        cursor = 2800

        for i in range(BATCH_SIZE):
            if i < active_users:
                hot = [cursor, cursor + 1, cursor + 2]
                cursor += 3
            else:
                hot = [42]
            per_user_hot.append(hot)

        return build_hot_logits(args, per_user_hot_tokens=per_user_hot), per_user_hot

    def test_small_batch_different_params(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits, per_user_hot = self._make_active_user_logits(args, active_users=2)
        params = SamplingParams(
            temperature=[0.0, 1.0],
            top_k=[1, 3],
            top_p=[1.0, 1.0],
        )
        tokens = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)[0]
        assert tokens[0] in set(per_user_hot[0])
        assert tokens[1] in set(per_user_hot[1])
        assert_tokens_in_vocab(tokens, args.vocab_size)

    def test_full_batch_different_params(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits, per_user_hot = self._make_active_user_logits(args, active_users=BATCH_SIZE)
        temperature = [0.5 + (i * 0.02) for i in range(BATCH_SIZE)]
        params = SamplingParams(temperature=temperature, top_k=[3] * BATCH_SIZE, top_p=[1.0] * BATCH_SIZE)
        tokens = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)[0]

        for i, tok in enumerate(tokens):
            assert tok in set(per_user_hot[i]), f"User {i} expected token from {per_user_hot[i]}, got {tok}"

    def test_partial_batch_different_params(self, mesh_device):
        args = make_sampling_args(mesh_device)
        active_users = BATCH_SIZE // 2
        logits, per_user_hot = self._make_active_user_logits(args, active_users=active_users)
        temperature = [0.0 if i % 2 == 0 else 1.0 for i in range(active_users)]
        top_k = [1 if i % 2 == 0 else 3 for i in range(active_users)]
        params = SamplingParams(temperature=temperature, top_k=top_k, top_p=[1.0] * active_users)
        tokens = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)[0]

        for i in range(active_users):
            assert tokens[i] in set(per_user_hot[i]), f"Active user {i} expected token from {per_user_hot[i]}"

        assert_tokens_in_vocab(tokens, args.vocab_size)


# --- Test: mixed-parameter batches ---


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
class TestMixedParameterBatches:
    def test_all_parameter_types_in_batch(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007])

        temperature_template = [0.0, 1.5, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0]
        top_k_template = [32, 8, 10, 8, 8, 8, 8, 8]
        top_p_template = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5]
        repetition_template = [1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.5, 1.0]
        presence_template = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0]
        frequency_template = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0]

        temperature = (temperature_template * 4)[:BATCH_SIZE]
        top_k = (top_k_template * 4)[:BATCH_SIZE]
        top_p = (top_p_template * 4)[:BATCH_SIZE]
        repetition = (repetition_template * 4)[:BATCH_SIZE]
        presence = (presence_template * 4)[:BATCH_SIZE]
        frequency = (frequency_template * 4)[:BATCH_SIZE]
        seeds = [12000 + i for i in range(BATCH_SIZE)]

        def _state_setup(sg):
            seen = torch.full((BATCH_SIZE, 1), 3000, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)

        params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition,
            presence_penalty=presence,
            frequency_penalty=frequency,
        )

        out1 = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=1,
            advance_seeds=True,
            seed_values=seeds,
            state_setup=_state_setup,
        )[0]
        out2 = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=1,
            advance_seeds=True,
            seed_values=seeds,
            state_setup=_state_setup,
        )[0]
        assert out1 == out2, "Mixed parameter batch should replay deterministically with fixed seeds"
        assert_tokens_in_vocab(out1, args.vocab_size)
        assert len(set(out1)) >= 2, "Mixed parameter batch should produce non-trivial diversity"


# --- Test: a single odd lane out (1 greedy vs 31 stochastic, and the mirror) ---


class TestSingleGreedyLaneInStochasticBatch:
    """One greedy user in an otherwise stochastic batch, and the mirror case.

    This is the configuration that stresses per-user sampling hardest, and it is
    not covered by uniformly-greedy or uniformly-stochastic batches:

    * The force-argmax fast path cannot fire, so the top-k pipeline has to honour
      a k=1 lane sitting next to 31 lanes that each draw from their own RNG
      stream. Every k/p/temp value is indexed per core, and the k=1 lane takes
      the ``k <= FACE_WIDTH`` branch of the writer kernel's candidate walk while
      its neighbours do not.
    * A batch that is *nearly* uniform is exactly where an "any lane" predicate
      passes for a "all lanes" rule -- see
      ``test_force_argmax_needs_every_lane_greedy``.

    Every user gets its own disjoint hot-token band so cross-lane leakage shows
    up as a concrete per-user failure rather than a diversity statistic.
    """

    BASE_TOKEN = 4000

    def _band_params(self, odd_lane, *, base_top_k, base_temperature, odd_top_k, odd_temperature, **overrides):
        """Uniform params for every lane, overridden on ``odd_lane`` only."""
        temperature = [base_temperature] * BATCH_SIZE
        top_k = [base_top_k] * BATCH_SIZE
        if odd_lane is not None:
            temperature[odd_lane] = odd_temperature
            top_k[odd_lane] = odd_top_k
        return SamplingParams(temperature=temperature, top_k=top_k, top_p=[1.0] * BATCH_SIZE, **overrides)

    def _greedy_lane_params(self, greedy_lane, *, temperature=1.0):
        """31 stochastic lanes plus (optionally) one greedy lane."""
        return self._band_params(
            greedy_lane,
            base_top_k=BAND_TOKENS_PER_USER,
            base_temperature=temperature,
            odd_top_k=1,
            odd_temperature=0.0,
        )

    def _assert_no_band_leakage(self, tokens, bands, *, context=""):
        for user, tok in enumerate(tokens):
            assert tok in set(bands[user]), (
                f"User {user} sampled token {tok} outside its own band "
                f"[{bands[user][0]}..{bands[user][-1]}]{context}. tokens={tokens}"
            )

    @pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
    @pytest.mark.parametrize(
        "device_params",
        RING_FABRIC_DEVICE_PARAMS,
        indirect=True,
    )
    @pytest.mark.parametrize("greedy_lane", ODD_LANE_INDICES_MULTI_DEVICE)
    def test_one_greedy_lane_rest_stochastic(self, mesh_device, device_params, greedy_lane):
        args = make_sampling_args(mesh_device)
        logits, bands = build_disjoint_band_logits(args, base_token=self.BASE_TOKEN)
        params = self._greedy_lane_params(greedy_lane)
        seeds = [40000 + i for i in range(BATCH_SIZE)]

        outputs = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=FAST_NUM_STEPS,
            advance_seeds=True,
            seed_values=seeds,
        )

        expected_greedy = bands[greedy_lane][0]
        for step, tokens in enumerate(outputs):
            assert_tokens_in_vocab(tokens, args.vocab_size)
            self._assert_no_band_leakage(tokens, bands, context=f" at step {step}")
            assert tokens[greedy_lane] == expected_greedy, (
                f"Greedy lane {greedy_lane} must pick its max-logit token {expected_greedy} "
                f"at every step, got {tokens[greedy_lane]} at step {step}. tokens={tokens}"
            )

        # The 31 stochastic lanes must still be sampling. If the k=1 lane dragged
        # the batch into argmax, every lane would return its own band rank 0.
        off_top_picks = sum(
            1 for tokens in outputs for user, tok in enumerate(tokens) if user != greedy_lane and tok != bands[user][0]
        )
        assert off_top_picks > 0, (
            f"Stochastic lanes never left their band's top token; the greedy lane "
            f"{greedy_lane} appears to have forced argmax on the whole batch. outputs={outputs}"
        )

        # Same seeds must replay bit-exactly, and every device view must agree.
        replay = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=FAST_NUM_STEPS,
            advance_seeds=True,
            seed_values=seeds,
        )
        assert replay == outputs, "Mixed greedy/stochastic batch must replay exactly for fixed seeds"

        for device_idx in representative_device_indices(mesh_device)[1:]:
            out_device = run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=FAST_NUM_STEPS,
                advance_seeds=True,
                seed_values=seeds,
                device_idx=device_idx,
            )
            assert out_device == outputs, f"Device view mismatch for device_idx={device_idx}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    @pytest.mark.parametrize("greedy_lane", ODD_LANE_INDICES)
    def test_greedy_lane_ignores_seeds(self, mesh_device, greedy_lane):
        """The greedy lane must be seed-independent while its neighbours are not."""
        args = make_sampling_args(mesh_device)
        logits, bands = build_disjoint_band_logits(args, base_token=self.BASE_TOKEN)
        params = self._greedy_lane_params(greedy_lane)

        seeds_a = [41000 + i for i in range(BATCH_SIZE)]
        seeds_b = [s + 987654 for s in seeds_a]
        out_a = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_a
        )[0]
        out_b = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_b
        )[0]

        expected_greedy = bands[greedy_lane][0]
        for label, tokens in (("seeds_a", out_a), ("seeds_b", out_b)):
            self._assert_no_band_leakage(tokens, bands, context=f" ({label})")
            assert tokens[greedy_lane] == expected_greedy, (
                f"Greedy lane {greedy_lane} changed with the seed vector ({label}): "
                f"expected {expected_greedy}, got {tokens[greedy_lane]}"
            )

        changed = [i for i in range(BATCH_SIZE) if i != greedy_lane and out_a[i] != out_b[i]]
        assert len(changed) >= 2, (
            "Shifting the seed vector should move several stochastic lanes; only "
            f"{changed} changed. out_a={out_a}, out_b={out_b}"
        )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    @pytest.mark.parametrize("greedy_lane", ODD_LANE_INDICES)
    def test_greedy_lane_does_not_perturb_other_lanes(self, mesh_device, greedy_lane):
        """Flipping one lane to k=1 must leave every other lane's token untouched.

        Each user is sampled on its own core with its own k/p/temp and its own
        rand tile, so lane ``greedy_lane`` going greedy is invisible to the other
        31 lanes. Any difference means per-user state is being shared.
        """
        args = make_sampling_args(mesh_device)
        logits, bands = build_disjoint_band_logits(args, base_token=self.BASE_TOKEN)
        seeds = [42000 + i for i in range(BATCH_SIZE)]

        # Only k differs between the two runs: temperature=0.0 is normalised to an
        # inverse temperature of 1.0, which is what temperature=1.0 maps to too.
        all_stochastic = self._greedy_lane_params(None)
        one_greedy = self._greedy_lane_params(greedy_lane)

        out_reference = run_sampling_generator(
            mesh_device, args, logits, all_stochastic, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        out_mixed = run_sampling_generator(
            mesh_device, args, logits, one_greedy, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]

        self._assert_no_band_leakage(out_reference, bands, context=" (all stochastic)")
        self._assert_no_band_leakage(out_mixed, bands, context=" (one greedy)")

        perturbed = [
            (user, out_reference[user], out_mixed[user])
            for user in range(BATCH_SIZE)
            if user != greedy_lane and out_reference[user] != out_mixed[user]
        ]
        assert not perturbed, (
            f"Making lane {greedy_lane} greedy changed other lanes (user, expected, got): {perturbed}. "
            f"reference={out_reference}, mixed={out_mixed}"
        )
        assert (
            out_mixed[greedy_lane] == bands[greedy_lane][0]
        ), f"Greedy lane {greedy_lane} should pick {bands[greedy_lane][0]}, got {out_mixed[greedy_lane]}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    @pytest.mark.parametrize("stochastic_lane", ODD_LANE_INDICES)
    def test_one_stochastic_lane_rest_greedy(self, mesh_device, stochastic_lane):
        """Mirror case: 31 greedy lanes must not silence the one sampling lane."""
        args = make_sampling_args(mesh_device)
        logits, bands = build_disjoint_band_logits(args, base_token=self.BASE_TOKEN)
        params = self._band_params(
            stochastic_lane,
            base_top_k=1,
            base_temperature=0.0,
            odd_top_k=BAND_TOKENS_PER_USER,
            odd_temperature=1.0,
        )
        # Enough steps that a genuinely stochastic lane repeating one token is
        # far less likely than any other cause of failure.
        num_steps = 2 * FAST_NUM_TRIES
        seeds = [43000 + i for i in range(BATCH_SIZE)]

        outputs = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=num_steps, advance_seeds=True, seed_values=seeds
        )

        for step, tokens in enumerate(outputs):
            assert_tokens_in_vocab(tokens, args.vocab_size)
            self._assert_no_band_leakage(tokens, bands, context=f" at step {step}")
            for user in range(BATCH_SIZE):
                if user == stochastic_lane:
                    continue
                assert tokens[user] == bands[user][0], (
                    f"Greedy lane {user} must pick {bands[user][0]} at every step, "
                    f"got {tokens[user]} at step {step}"
                )

        sampled = [step[stochastic_lane] for step in outputs]
        assert len(set(sampled)) >= 2, (
            f"Lane {stochastic_lane} was the only stochastic lane and never varied over "
            f"{num_steps} steps; the greedy majority appears to have forced it to argmax. "
            f"sampled={sampled}"
        )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    @pytest.mark.parametrize("greedy_lane", ODD_LANE_INDICES)
    def test_one_greedy_lane_with_penalties(self, mesh_device, greedy_lane):
        """A penalty on the greedy lane must retarget only that lane.

        The greedy lane's top band token is penalised below its runner-up, so the
        expected token is exact; the other 31 lanes carry no penalty and must be
        untouched by the penalty state written for their neighbour.
        """
        args = make_sampling_args(mesh_device)
        logits, bands = build_disjoint_band_logits(args, base_token=self.BASE_TOKEN)

        # top_logit=10.0 with step=0.25: dividing rank 0 by 4.0 gives 2.5, well
        # below rank 1 at 9.75, so the greedy pick moves by exactly one rank.
        repetition = [1.0] * BATCH_SIZE
        repetition[greedy_lane] = 4.0
        params = self._band_params(
            greedy_lane,
            base_top_k=BAND_TOKENS_PER_USER,
            base_temperature=1.0,
            odd_top_k=1,
            odd_temperature=0.0,
            repetition_penalty=repetition,
        )

        def _state_setup(sg):
            # -1 marks "no token seen", so only the greedy lane carries history.
            seen = torch.full((BATCH_SIZE, 1), -1, dtype=torch.int64)
            seen[greedy_lane, 0] = bands[greedy_lane][0]
            sg.reset_output_state(tokens=seen)

        seeds = [44000 + i for i in range(BATCH_SIZE)]
        tokens = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=1,
            advance_seeds=True,
            seed_values=seeds,
            state_setup=_state_setup,
        )[0]

        assert_tokens_in_vocab(tokens, args.vocab_size)
        self._assert_no_band_leakage(tokens, bands)
        assert tokens[greedy_lane] == bands[greedy_lane][1], (
            f"Penalised greedy lane {greedy_lane} should fall back to its runner-up "
            f"{bands[greedy_lane][1]}, got {tokens[greedy_lane]}. tokens={tokens}"
        )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    @pytest.mark.parametrize("odd_lane", ODD_LANE_INDICES)
    def test_force_argmax_needs_every_lane_greedy(self, mesh_device, odd_lane):
        """One non-greedy lane must keep the whole batch off the argmax fast path.

        ``TTSampling._is_force_argmax_sampling`` is an all-lanes predicate. If it
        ever degraded to "any lane", a single greedy user would silently force
        argmax on the 31 users who asked to sample -- and because the fast path
        skips the top-k/top-p/RNG pipeline entirely, nothing downstream would
        notice. This asserts the predicate directly, with the fast path enabled
        in ``model_config`` (the other tests leave it off).
        """
        args = make_sampling_args(mesh_device)
        args.model_config = {
            "SAMPLING_AG_CONFIG": {
                "allow_force_argmax": True,
                "num_links": 1,
                "topology": ttnn.Topology.Linear,
            }
        }
        logits, _ = build_disjoint_band_logits(args, base_token=self.BASE_TOKEN)

        observed = {}

        def _probe(label):
            def _record(sg):
                observed[label] = sg.tt_sampling.force_argmax_sampling

            return _record

        # num_steps=0 runs the host-side param plumbing without any sampling.
        all_greedy = self._band_params(None, base_top_k=1, base_temperature=0.0, odd_top_k=1, odd_temperature=0.0)
        run_sampling_generator(mesh_device, args, logits, all_greedy, num_steps=0, state_setup=_probe("all_greedy"))

        # temperature=1.0 also normalises to an inverse temperature of 1.0, so
        # only top_k distinguishes this lane -- the k branch of the predicate.
        one_stochastic = self._band_params(
            odd_lane,
            base_top_k=1,
            base_temperature=0.0,
            odd_top_k=BAND_TOKENS_PER_USER,
            odd_temperature=1.0,
        )
        run_sampling_generator(
            mesh_device, args, logits, one_stochastic, num_steps=0, state_setup=_probe("one_stochastic")
        )

        assert observed["all_greedy"] is True, "A fully greedy batch should take the force-argmax fast path"
        assert observed["one_stochastic"] is False, (
            f"Lane {odd_lane} requested top_k={BAND_TOKENS_PER_USER} but the batch still took the "
            "force-argmax fast path, which skips top-k/top-p/RNG entirely"
        )


# --- Test: traced sampling path ---

# Production decode calls SamplingGenerator.sample(..., enable_trace=True) -- that is the
# signature default. Every other test in this file runs enable_trace=False, so
# _trace_slot / capture_trace / _execute_trace / _validate_trace_inputs are otherwise
# never executed. Two things constrain what a traced test can look like:
#   * sample() computes `use_internal_trace = enable_trace and not
#     seed_manager.has_active_request_seed()`, so a test with explicit request seeds
#     silently falls back to the eager path and proves nothing. These use unseeded runs.
#   * trace replay binds the captured input address, so the same logits tensor object must
#     be reused on every step (run_sampling_generator does this when enable_trace=True) and
#     penalties -- which rewrite that tensor in place -- cannot be combined with it.
# A trace region has to be reserved at device open, hence the device_params override; the
# rest of the suite opens with none ("No trace region size for 1" in the CI log).
_TRACE_DEVICE_PARAMS = [{"trace_region_size": 23887872}]


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", _TRACE_DEVICE_PARAMS, indirect=True)
class TestTracedSampling:
    def test_traced_greedy_matches_untraced(self, mesh_device, device_params):
        """Capture-then-replay must return exactly what the eager path returns.

        Greedy (temperature=0.0, top_k=1) makes the expected token independent of RNG state,
        so the traced result is directly comparable to the untraced one across replays --
        step 1 captures, steps 2 and 3 go through _execute_trace.
        """
        args = make_sampling_args(mesh_device)
        hot_tokens = [3300, 3301, 3302, 3303]
        logits = build_hot_logits(args, hot_tokens=hot_tokens)
        params = SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[1] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
        )

        untraced = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, enable_trace=False
        )[0]
        traced = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=3, advance_seeds=True, enable_trace=True
        )

        assert all(
            tok == hot_tokens[0] for tok in untraced
        ), f"Eager greedy should pick the max-logit token {hot_tokens[0]}, got {untraced}"
        for step, tokens in enumerate(traced):
            assert tokens == untraced, (
                f"Traced step {step} diverged from the eager result. " f"traced={tokens}, untraced={untraced}"
            )

    def test_trace_rejects_mismatched_logits_tensor(self, mesh_device, device_params, expect_error):
        """_validate_trace_inputs must reject a logits tensor the trace was not captured on.

        Replay reuses the captured input buffer, so silently accepting a different tensor
        would sample stale logits. Exercised directly because run_sampling_generator always
        reuses one tensor by construction.
        """
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[3400, 3401, 3402])
        padded = pad_logits_to_max_batch(logits, max_batch_size=BATCH_SIZE)
        params = SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[1] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
        )

        sg = None
        captured_input = None
        other_input = None
        try:
            sg = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None)
            sg.reset_sampling_params(format_sampling_params(params, BATCH_SIZE))
            # Unseeded, or sample() would bypass the trace entirely.
            sg.seed_manager.reset_seed(None, list(range(BATCH_SIZE)))
            sg.seed_manager.get_new_values()

            captured_input = make_sharded_logits(padded, mesh_device, args)
            sg.sample(captured_input, enable_trace=True)  # captures
            ttnn.synchronize_device(mesh_device)

            other_input = make_sharded_logits(padded, mesh_device, args)
            with expect_error(ValueError, r"does not match the tensor used during trace capture"):
                sg.sample(other_input, enable_trace=True)
        finally:
            if sg is not None:
                try:
                    sg.reset_trace()
                except Exception:
                    pass
            for tensor in (captured_input, other_input):
                if tensor is not None:
                    del tensor
            if sg is not None:
                del sg
            safe_sync(mesh_device)


# --- Test: format_sampling_params lane semantics (host-only, no device) ---


# These need no device and no mesh_device fixture. They belong beside the other host-only
# sampling tests in models/common/tests/test_sampling.py, but that file is not wired into
# any pipeline today, so they live here to actually run. Move them when it is. (#38316)
class TestFormatSamplingParamsLanes:
    """Pin the per-lane contract of format_sampling_params.

    A scalar companion field alongside a per-user temperature must reach every active lane.
    Getting this wrong is invisible: the padding defaults are top_k=1 (greedy), top_p=1.0
    and penalty no-ops, so an under-broadcast field silently changes how real lanes sample
    while every assertion about token ranges still passes.
    """

    @staticmethod
    def _fmt(**kwargs):
        return format_sampling_params(SamplingParams(**kwargs), BATCH_SIZE)

    def test_all_scalar_input_is_lane_zero_plus_greedy_padding(self):
        """The historical all-scalar shape must be untouched by the broadcast rule."""
        out = self._fmt(temperature=0.5, top_k=8, top_p=0.9)
        assert out.temperature[0] == pytest.approx(1 / 0.5)  # inverted for the device
        assert out.top_k[0] == 8
        assert out.top_p[0] == pytest.approx(0.9)
        # Inactive lanes are padded to the greedy representation (temp 1.0 / k 1 / p 0.0).
        for lane in range(1, BATCH_SIZE):
            assert out.top_k[lane] == 1, f"lane {lane}"
            assert out.temperature[lane] == pytest.approx(1.0), f"lane {lane}"

    def test_scalar_companion_broadcasts_across_active_lanes(self):
        """A scalar top_k/top_p must not leave active lanes on the greedy default."""
        active = 8
        out = self._fmt(temperature=[0.8] * active, top_k=8, top_p=0.9)
        for lane in range(active):
            assert out.top_k[lane] == 8, f"active lane {lane} fell back to the k=1 default"
            assert out.top_p[lane] == pytest.approx(0.9), f"active lane {lane}"
            assert out.temperature[lane] == pytest.approx(1 / 0.8), f"active lane {lane}"
        for lane in range(active, BATCH_SIZE):
            assert out.top_k[lane] == 1, f"inactive lane {lane} should be greedy"

    def test_single_element_companion_stays_lane_scoped(self):
        """[x] keeps targeting lane 0; reinterpreting it as a broadcast would change
        sampling for existing callers' other lanes."""
        out = self._fmt(temperature=[1.0, 1.0], top_k=[8], top_p=1.0)
        assert out.top_k[0] == 8
        assert out.top_k[1] == 1, "a 1-element list must not broadcast onto lane 1"

    def test_mismatched_companion_length_is_rejected(self):
        """Silently padding the gap would turn real lanes greedy."""
        try:
            self._fmt(temperature=[1.0] * 8, top_k=[3, 3, 3], top_p=1.0)
        except ValueError as exc:
            assert "top_k" in str(exc) and "8 active lanes" in str(exc)
        else:
            raise AssertionError("a top_k list shorter than the active lanes must be rejected")

    def test_scalar_penalty_reaches_every_active_lane(self):
        """Regression: a scalar penalty used to land on lane 0 only, leaving the rest on the
        no-op default with no diagnostic."""
        active = 8
        out = self._fmt(
            temperature=[1.0] * active,
            top_k=8,
            top_p=1.0,
            repetition_penalty=1.2,
            presence_penalty=0.5,
            frequency_penalty=0.25,
        )
        for lane in range(active):
            assert out.repetition_penalty[lane] == pytest.approx(1.2), f"active lane {lane}"
            assert out.presence_penalty[lane] == pytest.approx(0.5), f"active lane {lane}"
            assert out.frequency_penalty[lane] == pytest.approx(0.25), f"active lane {lane}"
        for lane in range(active, BATCH_SIZE):
            assert out.repetition_penalty[lane] == pytest.approx(1.0), f"inactive lane {lane}"
            assert out.presence_penalty[lane] == pytest.approx(0.0), f"inactive lane {lane}"

    def test_unset_penalties_are_all_no_ops(self):
        """The SamplingParams penalty defaults are scalars equal to the padding defaults, so
        a caller who never sets them must be unaffected by the broadcast rule."""
        out = self._fmt(temperature=[1.0] * 8, top_k=8, top_p=1.0)
        assert all(v == pytest.approx(1.0) for v in out.repetition_penalty)
        assert all(v == pytest.approx(0.0) for v in out.presence_penalty)
        assert all(v == pytest.approx(0.0) for v in out.frequency_penalty)

    def test_scalar_seed_stays_on_lane_zero(self):
        """seed is deliberately NOT broadcast: one seed on every lane means every lane draws
        the same token, which a caller has to ask for explicitly."""
        out = self._fmt(temperature=[1.0] * 8, top_k=8, top_p=1.0, seed=1234)
        assert out.seed[0] == 1234
        assert all(s is None for s in out.seed[1:]), f"a scalar seed must not broadcast: {out.seed[:4]}"

    def test_log_probs_scalar_broadcasts_to_all_lanes(self):
        """enable_log_probs selects an output format rather than shaping a distribution, so
        it broadcasts to max_batch_size, not to the active lane count."""
        out = self._fmt(temperature=[1.0, 1.0], top_k=8, top_p=1.0, enable_log_probs=True)
        assert all(out.enable_log_probs), "enable_log_probs should cover every lane"
