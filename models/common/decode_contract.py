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
    Upload temperature/top-k/top-p/penalty parameters, **and register each request's
    explicit seed**. The seed half is easy to miss from the name: parameters carry
    seeds, so an adapter that uploads only k/p/temperature leaves the seed manager
    unseeded and two requests sharing a seed then diverge. Registration here is
    conditional (it compares requested against cached seeds), which is why the first
    decode of a lifecycle needs ``reset_sampling_state`` as well.
``reset_sampling_state``
    Rebuild per-slot sampling state: prompt/output penalty history and seeds. The
    seed reset is unconditional, including ``seed=None``, so a decode-only sampling
    lifecycle gets a fresh device seed.

``slot_remap`` accompanies the commands on every contract decode, including
host-sampling steps, and must be applied to every persistent slot-indexed state the
forward reads (sampler seeds/RNG, recurrent state, RoPE deltas) exactly once.

An adapter advertises support with ``decode_input_update_contract = 1``. Adapters
that cannot execute part of the contract reject those combinations loudly rather
than silently degrading; see ``require_full_input_reload``. Doing so is conformant
only while the adapter leaves ``model_capabilities["supports_async_decode"]`` off,
which is what stops vLLM from planning the combination it cannot run.

Declare the version on the generator that implements the commands, not on each leaf
adapter, so a subclass cannot silently drop to the legacy call shape. Every subclass
therefore inherits the assertion, and one that overrides ``decode_forward`` must
re-establish it for itself.

vLLM sends all four commands on every contract decode, so an adapter needs no
defaults. Where it declares one, the only safe value is the host-authoritative one:
``reload_inputs=True`` and the other three ``False``. Any other default silently
reuses device state the caller did not ask to keep.
"""

from enum import Enum


class DecodeInputStaging(str, Enum):
    """What a decode step must copy into its trace inputs.

    The two input commands are not independent switches, so a generator branches on
    this instead of on the pair. Reading them as independent looks reasonable and is
    wrong: ``reload_inputs=True, reload_page_table=False`` is vLLM's every-transition
    shape, and an adapter whose page-table copy hangs off ``reload_page_table`` alone
    would then never refresh it, leaving the device addressing the previous batch's KV
    blocks with no error to show for it.
    """

    ALL = "all"
    PAGE_TABLE_ONLY = "page_table_only"
    NONE = "none"


def decode_input_staging(reload_inputs: bool, reload_page_table: bool) -> DecodeInputStaging:
    """Collapse the two input commands into the one decision they encode."""
    if reload_inputs:
        if reload_page_table:
            raise ValueError(
                "reload_page_table is the page-table-only copy and is meaningless with "
                "reload_inputs=True, which already restages page tables"
            )
        return DecodeInputStaging.ALL
    if reload_page_table:
        return DecodeInputStaging.PAGE_TABLE_ONLY
    return DecodeInputStaging.NONE


def reject_legacy_reload_signal(adapter: str, kwargs: dict) -> None:
    """Reject the pre-contract ``reset_batch`` keyword by name.

    A caller old enough to send it also omits the four commands, so its layout changes
    would silently take the host-authoritative defaults and be reinterpreted as a full
    reload on every step. Failing here names the fix instead.
    """
    if "reset_batch" in kwargs:
        raise ValueError(
            f"reset_batch is not part of the decode input-update contract; the vLLM build "
            f"driving {adapter} predates it. Upgrade the vLLM pin to a build that sends "
            f"reload_inputs / reload_page_table / reload_sampling_params / "
            f"reset_sampling_state."
        )


def per_layer_page_tables_need_upload(commands: dict) -> bool:
    """Whether a hybrid-attention model must re-upload its per-layer page tables.

    Takes the whole command mapping so the omitted-command defaults are decided here
    rather than restated at each call site: three adapters ask this question, and
    defaults spelled out per site have to agree by hand. They match the defaults
    ``decode_forward`` declares, so an omitting caller gets the conservative full
    refresh at both layers.
    """
    staging = decode_input_staging(commands.get("reload_inputs", True), commands.get("reload_page_table", False))
    return staging is not DecodeInputStaging.NONE


def rank_local_slot_remap(slot_remap, rank: int, slots_per_rank: int, data_parallel: int = 1) -> list[int]:
    """Rebase one DP rank's slice of a global ``slot_remap`` onto rank-local slots.

    ``slot_remap`` holds GLOBAL slot indices: vLLM offsets each rank's local
    ``[0, slots_per_rank)`` mapping by ``rank * slots_per_rank``. Per-rank state such as
    the seed manager indexes its own rank-local slots, so the slice has to be rebased
    before it is handed over, or rank >= 1 indexes past the end of its own state.

    The length check is the actual cross-rank guard. A range check on each value is not:
    if the caller's ``slots_per_rank`` is larger than the stride vLLM used, rank 0's slice
    silently swallows rank 1's entries, every one of them lands inside
    ``[0, slots_per_rank)``, and rank 1 gets an empty slice. Both sides must agree on the
    stride, so require the mapping to be exactly as wide as this rank's view implies.
    """
    expected = data_parallel * slots_per_rank
    if len(slot_remap) != expected:
        raise ValueError(
            f"slot_remap has {len(slot_remap)} entries but this generator expects "
            f"{expected} ({data_parallel} DP rank(s) x {slots_per_rank} slots); the "
            f"plugin and the model disagree on the per-rank slot stride"
        )
    base = rank * slots_per_rank
    rank_remap = []
    for new_slot, old_value in enumerate(slot_remap[base : base + slots_per_rank]):
        old_slot = int(old_value) - base
        if not 0 <= old_slot < slots_per_rank:
            raise ValueError(
                f"slot_remap[{base + new_slot}]={int(old_value)} moves a request across "
                f"DP rank {rank} (slots [{base}, {base + slots_per_rank}))"
            )
        rank_remap.append(old_slot)
    return rank_remap


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
