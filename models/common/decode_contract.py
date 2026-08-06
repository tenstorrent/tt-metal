# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Decode input-update contract shared by the vLLM-facing generators.

vLLM owns decode reload policy; a generator executes the commands it is given and
adds no page-table comparisons, sampling-mode inference, or forced reloads of its
own. Every decode call carries four independent commands:

``reload_inputs``
    Host token/position/page-table inputs are authoritative for this step; restage
    all of them. Subsumes ``reload_page_table``.
``reload_page_table``
    Copy only the page-table trace input, preserving device-produced token and
    position state. Meaningful only when ``reload_inputs`` is false.
``reload_sampling_params``
    Upload temperature/top-k/top-p/penalty parameters.
``reset_sampling_state``
    Rebuild per-slot sampling state: prompt/output penalty history and seeds. The
    seed reset is unconditional, including ``seed=None``, so a decode-only sampling
    lifecycle gets a fresh device seed.

``slot_remap`` accompanies the commands on every contract decode, including
host-sampling steps, and must be applied to every persistent slot-indexed state the
forward reads (sampler seeds/RNG, recurrent state, RoPE deltas) exactly once.

An adapter advertises support with ``decode_input_update_contract = 1``. Adapters
that cannot execute part of the contract reject those combinations loudly rather
than silently degrading; see ``require_full_input_reload``.
"""


def per_layer_page_tables_need_upload(reload_inputs: bool = True, reload_page_table: bool = False) -> bool:
    """Whether a hybrid-attention model must re-upload its per-layer page tables.

    Defaults are the conservative full refresh: a caller that omits the commands must
    fail in ``decode_forward``, which names them, not here.
    """
    return bool(reload_inputs or reload_page_table)


def require_full_input_reload(
    adapter: str,
    *,
    reload_inputs: bool,
    reload_page_table: bool,
    reload_sampling_params: bool,
    reset_sampling_state: bool,
) -> None:
    """Reject commands a host-authoritative adapter with no device sampler cannot run.

    Names the offending commands: a bare "requires a full host-input reload" does not
    tell the caller which of the four it got wrong.
    """
    rejected = []
    if not reload_inputs:
        rejected.append("reload_inputs=False")
    if reload_page_table:
        rejected.append("reload_page_table=True")
    if reload_sampling_params:
        rejected.append("reload_sampling_params=True")
    if reset_sampling_state:
        rejected.append("reset_sampling_state=True")
    if rejected:
        raise ValueError(
            f"{adapter} decode accepts only a full host-input reload and holds no device "
            f"sampling state, but got {', '.join(rejected)}"
        )
