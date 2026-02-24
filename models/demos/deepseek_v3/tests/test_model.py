# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import errno
import hashlib
import re
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN, build_test_cases_and_ids
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    dequantize_state_dict,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_caches_from_torch,
    run_reference_with_attention,
    torch_cache_from_transformers,
    transformers_cache_from_torch,
)

REFERENCE_OUTPUT_CACHE_FILE_PREFIX = "deepseek_v3_model_reference_outputs"
REFERENCE_OUTPUT_CACHE_LEGACY_FILENAME = f"{REFERENCE_OUTPUT_CACHE_FILE_PREFIX}.pt"
PCC_REQUIRED_PREFILL = 0.97
PCC_REQUIRED_DECODE = 0.97
REFERENCE_ENTRY_VERSION = 1


def _default_reference_cache_dir(cache_path: Path) -> Path:
    return cache_path / "tests_cache"


def _legacy_reference_cache_path(cache_path: Path) -> Path:
    return _default_reference_cache_dir(cache_path) / REFERENCE_OUTPUT_CACHE_LEGACY_FILENAME


def _build_case_identity(
    *,
    mode: str,
    seq_len: int,
    batch_size_per_row: int,
    mesh_shape: tuple[int, int],
    decode_position_ids: int | None,
    hf_config: PretrainedConfig,
) -> dict[str, str | int]:
    return {
        "mode": mode,
        "seq": int(seq_len),
        "batch_per_row": int(batch_size_per_row),
        "mesh": f"{mesh_shape[0]}x{mesh_shape[1]}",
        "decode_pos": "auto" if decode_position_ids is None else str(decode_position_ids),
        "layers": int(hf_config.num_hidden_layers),
        "max_seq": int(hf_config.max_seq_len),
    }


def _case_reference_cache_filename(case_identity: dict[str, str | int]) -> str:
    # Primary filename format: deterministic, human-readable, and hash-free.
    return (
        f"{REFERENCE_OUTPUT_CACHE_FILE_PREFIX}."
        f"mode_{case_identity['mode']}_seq_{case_identity['seq']}_batch_per_row_{case_identity['batch_per_row']}_"
        f"mesh_{case_identity['mesh']}_decode_pos_{case_identity['decode_pos']}_layers_{case_identity['layers']}_"
        f"max_seq_{case_identity['max_seq']}.pt"
    )


def _case_reference_cache_path(cache_path: Path, case_identity: dict[str, str | int]) -> Path:
    return _default_reference_cache_dir(cache_path) / _case_reference_cache_filename(case_identity)


def _legacy_case_reference_cache_filename(case_key: str) -> str:
    # Keep filenames deterministic and readable while avoiding path-unsafe characters.
    normalized_case_key = re.sub(r"[^a-zA-Z0-9]+", "_", case_key).strip("_").lower() or "case"
    digest = hashlib.sha1(case_key.encode("utf-8")).hexdigest()[:12]
    return f"{REFERENCE_OUTPUT_CACHE_FILE_PREFIX}.{normalized_case_key[:96]}.{digest}.pt"


def _legacy_case_reference_cache_path(cache_path: Path, case_key: str) -> Path:
    return _default_reference_cache_dir(cache_path) / _legacy_case_reference_cache_filename(case_key)


def _build_case_key(
    *,
    mode: str,
    seq_len: int,
    batch_size_per_row: int,
    mesh_shape: tuple[int, int],
    decode_position_ids: int | None,
    hf_config: PretrainedConfig,
) -> str:
    decode_pos = "auto" if decode_position_ids is None else str(decode_position_ids)
    return (
        f"mode={mode}|seq={seq_len}|batch_per_row={batch_size_per_row}|mesh={mesh_shape[0]}x{mesh_shape[1]}"
        f"|decode_pos={decode_pos}|layers={hf_config.num_hidden_layers}|max_seq={hf_config.max_seq_len}"
    )


def _extract_tt_logits_full(
    tt_output: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
) -> torch.Tensor:
    tt_output_shape = tuple(tt_output.shape)
    if len(tt_output_shape) != 4:
        raise ValueError(f"Expected 4D TT output tensor, got shape {tt_output_shape}")

    full_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
    )
    full_torch = full_torch.cpu().float()
    vocab = full_torch.shape[-1]
    return full_torch.reshape(1, -1, vocab)


def _load_torch_payload(path: Path) -> object:
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def _load_legacy_reference_cache(path: Path) -> dict:
    if not path.is_file():
        return {"version": 1, "cases": {}}

    payload = _load_torch_payload(path)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cache format in {path}: expected dict, got {type(payload)}")
    payload.setdefault("version", 1)
    payload.setdefault("cases", {})
    if not isinstance(payload["cases"], dict):
        raise ValueError(f"Invalid cache format in {path}: 'cases' must be a dict")
    return payload


def _load_case_reference_entry(path: Path) -> dict | None:
    if not path.is_file():
        return None

    payload = _load_torch_payload(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cache format in {path}: expected dict, got {type(payload)}")
    return payload


def _save_case_reference_entry(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _try_save_case_reference_entry(path: Path, payload: dict, *, reason: str) -> bool:
    try:
        _save_case_reference_entry(path, payload)
        return True
    except OSError as exc:
        if exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            logger.warning(
                f"Unable to persist reference cache ({reason}) to {path}: {exc}. "
                "Continuing with loaded/generated in-memory reference entry."
            )
            return False
        raise


def _expected_reference_output_shape(token_count: int, hf_config: PretrainedConfig) -> tuple[int, int, int]:
    return (1, token_count, hf_config.vocab_size)


def _generate_reference_case_entry(
    *,
    mode: str,
    seq_len: int,
    batch_size: int,
    hf_config: PretrainedConfig,
    state_dict: dict[str, torch.Tensor],
    position_ids: torch.Tensor | None,
    torch_input: torch.Tensor,
) -> dict:
    logger.info(f"Generating missing reference output (mode={mode}, seq_len={seq_len}, batch_size={batch_size})")
    with torch.device("meta"):
        reference_model = DeepseekV3ForCausalLM(hf_config).eval()
    reference_model = reference_model.to_empty(device=torch.device("cpu"))
    reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))
    reference_model = reference_model.to(torch.bfloat16)

    decode_input_caches = None
    if mode == "decode":
        prefill_len = int(position_ids.max().item()) if position_ids is not None else 0
        cache_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
        if prefill_len > 0:
            prefill_input = torch.randint(0, hf_config.vocab_size - 1, (batch_size, prefill_len), dtype=torch.long)
            prefill_seq_lens = torch.full((batch_size,), prefill_len, dtype=torch.long)
            _, _, prefill_cache = run_reference_with_attention(
                reference_model.model,
                prefill_input,
                prefill_seq_lens,
                None,
                hf_config,
                "prefill",
                False,
                collect_output=False,
            )
            decode_input_caches = torch_cache_from_transformers(prefill_cache)
        else:
            decode_input_caches = tuple(
                torch.empty((batch_size, 1, 0, cache_dim), dtype=torch.bfloat16)
                for _ in range(hf_config.num_hidden_layers)
            )

        torch_input_batch_first = torch_input.transpose(1, 0).contiguous()
        position_ids_2d = position_ids.unsqueeze(1)
        max_position_id = int(position_ids.max().item())
        mask = torch.full((batch_size, 1, 1, max_position_id + 1), float("-inf"), dtype=torch.bfloat16)
        for mask_row, position_id in zip(mask, position_ids):
            mask_row[:, :, :position_id] = 0.0
        mask[:, :, :, -1] = 0.0

        with torch.no_grad():
            model_output = reference_model(
                torch_input_batch_first,
                attention_mask=mask,
                position_ids=position_ids_2d,
                output_attentions=False,
                use_cache=True,
                past_key_values=transformers_cache_from_torch(decode_input_caches),
            )
        reference_output = model_output.logits.transpose(1, 0).float().cpu()
    else:
        position_ids_or_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
        hidden_states, _, _ = run_reference_with_attention(
            reference_model.model, torch_input, position_ids_or_seq_lens, None, hf_config, mode, False
        )
        with torch.no_grad():
            reference_output = reference_model.lm_head(hidden_states.to(torch.bfloat16)).float().cpu()
        reference_output = reference_output.reshape(1, -1, reference_output.shape[-1])

    return {
        "entry_version": REFERENCE_ENTRY_VERSION,
        "source": "reference",
        "reference_output": reference_output,
        "decode_input_caches": list(decode_input_caches) if mode == "decode" else None,
    }


def generate_io_without_reference(
    mode: str,
    seq_len: int,
    batch_size: int,
    hf_config: PretrainedConfig,
    decode_position_id: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    torch_input = torch.randint(0, hf_config.vocab_size - 1, (batch_size, seq_len), dtype=torch.long)
    position_ids = None
    if mode == "decode":
        if decode_position_id is None:
            position_ids = torch.randint(0, hf_config.max_seq_len - 1, (batch_size,), dtype=torch.long)
        else:
            if not isinstance(decode_position_id, int):
                raise ValueError(f"decode_position_id must be int or None, got {type(decode_position_id)}")
            if not (0 <= decode_position_id < hf_config.max_seq_len):
                raise ValueError(
                    f"decode_position_id must be in [0, {hf_config.max_seq_len - 1}], got {decode_position_id}"
                )
            position_ids = torch.full((batch_size,), decode_position_id, dtype=torch.long)
        torch_input = torch_input.transpose(1, 0)
    return position_ids, torch_input


def run_test_forward_pass_dpmodel(
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    state_dict,
    decode_position_ids: int | None = None,
):
    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports a batch size of 1"
        batch_size = batch_size_per_row
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"
        batch_size = batch_size_per_row * mesh_device.shape[0]

    state_dict = sub_state_dict(state_dict, "", hf_config_short.num_hidden_layers)

    case_key = _build_case_key(
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=batch_size_per_row,
        mesh_shape=tuple(mesh_device.shape),
        decode_position_ids=decode_position_ids,
        hf_config=hf_config_short,
    )
    case_identity = _build_case_identity(
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=batch_size_per_row,
        mesh_shape=tuple(mesh_device.shape),
        decode_position_ids=decode_position_ids,
        hf_config=hf_config_short,
    )

    logger.info("Setting up test IO (no-reference runtime)")
    position_ids, torch_input = generate_io_without_reference(
        mode, seq_len, batch_size, hf_config_short, decode_position_ids
    )
    expected_global_token_count = seq_len if mode == "prefill" else batch_size
    expected_local_token_count = expected_global_token_count // mesh_device.shape[0]
    expected_reference_output_shape = _expected_reference_output_shape(expected_global_token_count, hf_config_short)

    cache_file = _case_reference_cache_path(cache_path, case_identity)
    cached_case = _load_case_reference_entry(cache_file)
    if cached_case is None:
        legacy_case_file = _legacy_case_reference_cache_path(cache_path, case_key)
        if legacy_case_file != cache_file:
            cached_case = _load_case_reference_entry(legacy_case_file)
            if isinstance(cached_case, dict):
                if _try_save_case_reference_entry(cache_file, cached_case, reason="legacy per-case migration"):
                    logger.info(
                        f"Migrated legacy per-case reference baseline for case '{case_key}' "
                        f"from {legacy_case_file} to {cache_file}"
                    )
    if cached_case is None:
        legacy_cache = _load_legacy_reference_cache(_legacy_reference_cache_path(cache_path))
        cached_case = legacy_cache["cases"].get(case_key)
        if isinstance(cached_case, dict):
            if _try_save_case_reference_entry(cache_file, cached_case, reason="legacy monolithic migration"):
                logger.info(f"Migrated legacy reference baseline for case '{case_key}' to {cache_file}")
    cached_reference_output = cached_case.get("reference_output") if isinstance(cached_case, dict) else None
    cached_shape = tuple(cached_reference_output.shape) if isinstance(cached_reference_output, torch.Tensor) else None
    needs_regen = (
        not isinstance(cached_case, dict)
        or cached_case.get("entry_version") != REFERENCE_ENTRY_VERSION
        or cached_case.get("source") != "reference"
        or not isinstance(cached_reference_output, torch.Tensor)
        or cached_shape != expected_reference_output_shape
        or (mode == "decode" and cached_case.get("decode_input_caches") is None)
    )
    if needs_regen:
        logger.warning(f"Reference cache miss for case '{case_key}'. Generating reference output.")
        cached_case = _generate_reference_case_entry(
            mode=mode,
            seq_len=seq_len,
            batch_size=batch_size,
            hf_config=hf_config_short,
            state_dict=state_dict,
            position_ids=position_ids,
            torch_input=torch_input,
        )
        if _try_save_case_reference_entry(cache_file, cached_case, reason="newly generated case"):
            logger.info(f"Wrote reference baseline for case '{case_key}' to {cache_file}")
    else:
        logger.info(f"Using cached reference baseline for case '{case_key}' from {cache_file}")

    logger.info("Setting up model configs")
    mesh_rows, dp_factor = mesh_device.shape
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW, ()).item()
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, dp_factor)

    if mode == "decode":
        decode_input_caches = cached_case.get("decode_input_caches")
        if decode_input_caches is None:
            pytest.fail(f"Missing decode_input_caches in reference baseline for case '{case_key}'")
        if not isinstance(decode_input_caches, tuple):
            decode_input_caches = tuple(decode_input_caches)

        denom = mesh_rows * dp_factor
        assert batch_size % denom == 0, f"batch_size={batch_size} not divisible by mesh_rows*dp_factor={denom}"
        batches_per_device = batch_size // denom

        assert (
            paged_config.max_num_blocks % batches_per_device == 0
        ), f"max_num_blocks={paged_config.max_num_blocks} not divisible by batches_per_device={batches_per_device}"
        blocks_per_batch = paged_config.max_num_blocks // batches_per_device

        mapping = torch.arange(batches_per_device * blocks_per_batch, dtype=torch.long).reshape(
            batches_per_device, blocks_per_batch
        )
        mappings = tuple(mapping for _ in range(hf_config_short.num_hidden_layers))
        paged_input_caches, torch_page_tables = paged_caches_from_torch(
            decode_input_caches, tuple(mesh_device.shape), paged_config, user_id=None, mappings=mappings
        )
        tt_page_tables = tuple(
            MLA2D.create_page_table(page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device)
            for torch_page_table in torch_page_tables
        )
    else:
        paged_input_caches = None
        total_global_users = mesh_rows * USERS_PER_ROW
        num_devices = mesh_rows * dp_factor

        assert (
            total_global_users % num_devices == 0
        ), f"total_global_users={total_global_users} not divisible by num_devices={num_devices}"
        batches_per_device = total_global_users // num_devices

        assert (
            paged_config.max_num_blocks % batches_per_device == 0
        ), f"max_num_blocks={paged_config.max_num_blocks} not divisible by batches_per_device={batches_per_device}"
        blocks_per_batch = paged_config.max_num_blocks // batches_per_device

        torch_page_table = torch.arange(paged_config.max_num_blocks, dtype=torch.int32).reshape(
            batches_per_device, blocks_per_batch
        )
        tt_page_tables = tuple(
            MLA2D.create_page_table(page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device)
            for _ in range(hf_config_short.num_hidden_layers)
        )

    weight_config = get_test_weight_config(
        RowBatchedModel,
        hf_config_short,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
        test_name="test_model",
        real_weights=True,
    )
    model_config = get_model_config(RowBatchedModel, mode, hf_config_short, mesh_device)
    model_state = RowBatchedModel.create_state(hf_config_short, paged_config, mesh_device, ccl, paged_input_caches)
    model_shared_state = RowBatchedModel.create_shared_state(hf_config_short, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    logger.info("Setting up model inputs")
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    position_ids_tensor = (
        ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            dtype=ttnn.int32,
        )
        if mode == "decode"
        else None
    )

    rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)

    logger.info("Running TTNN forward pass")
    if mode == "prefill":
        tt_output = RowBatchedModel.forward_prefill(tt_input, user_id, run_config, rope_tensors, tt_page_tables)
    else:
        tt_output = RowBatchedModel.forward_decode(
            tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_tables
        )

    ttnn.synchronize_device(mesh_device)

    tt_output_shape = tuple(tt_output.shape)
    global_token_count = tt_output_shape[2] * mesh_device.shape[0]
    local_token_count = tt_output_shape[2]
    assert (
        global_token_count == expected_global_token_count
    ), f"Unexpected global token count: got {global_token_count}, expected {expected_global_token_count}"
    assert (
        local_token_count == expected_local_token_count
    ), f"Unexpected local token count: got {local_token_count}, expected {expected_local_token_count}"

    tt_output_torch = _extract_tt_logits_full(tt_output, mesh_device)
    assert (
        tuple(tt_output_torch.shape) == expected_reference_output_shape
    ), f"Unexpected TT output shape: got {tuple(tt_output_torch.shape)}, expected {expected_reference_output_shape}"
    assert torch.isfinite(tt_output_torch).all(), "Detected inf/nan in TT full output"

    expected_output = cached_case["reference_output"]
    if not isinstance(expected_output, torch.Tensor):
        expected_output = torch.tensor(expected_output)
    expected_output = expected_output.cpu().float()
    assert tuple(expected_output.shape) == expected_reference_output_shape, (
        f"Unexpected cached reference output shape: got {tuple(expected_output.shape)}, "
        f"expected {expected_reference_output_shape}"
    )

    pcc_required = PCC_REQUIRED_DECODE if mode == "decode" else PCC_REQUIRED_PREFILL
    assert_hidden_dim_pcc(tt_output_torch, expected_output, pcc_required=pcc_required)
    logger.info(f"TT full-output check passed against reference baseline for case '{case_key}'")

    ttnn.deallocate(tt_output)


TEST_CASES, TEST_IDS = build_test_cases_and_ids(
    USERS_PER_ROW,
    DEFAULT_PREFILL_SEQ_LEN,
)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.timeout(10000)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row, decode_position_ids",
    TEST_CASES,
    ids=TEST_IDS,
)
def test_forward_pass(
    mode,
    seq_len,
    batch_size_per_row,
    decode_position_ids,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    hf_config_short.num_hidden_layers = 5

    if mode != "decode":
        decode_position_ids = None

    run_test_forward_pass_dpmodel(
        mode,
        seq_len,
        batch_size_per_row,
        hf_config_short,
        cache_path,
        mesh_device,
        ccl,
        force_recalculate_weight_config,
        state_dict,
        decode_position_ids,
    )


if __name__ == "__main__":
    pytest.main([__file__])
