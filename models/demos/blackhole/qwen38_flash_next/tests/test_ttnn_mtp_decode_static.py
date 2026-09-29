# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_decode import (
    MAX_CONTEXT,
    Qwen38FixedFiveTarget,
    Qwen38MTPAlignmentCommit,
    Qwen38MTPAlignmentStep,
    Qwen38MTPDraftBatch,
    Qwen38MTPDraftEngine,
    Qwen38MTPDraftStep,
    Qwen38MTPQSASelectionProof,
    Qwen38MTPSeed,
    Qwen38SpeculativeDecodeController,
    Qwen38SpeculativeDecodePoisonedError,
    Qwen38SpeculativeDecodeUnavailableError,
    Qwen38SpeculativeStatus,
    Qwen38SpeculativeStopReason,
    Qwen38StateRollback,
    Qwen38TargetCommit,
    Qwen38TargetCommitMode,
    Qwen38TargetVerification,
    resolve_greedy_five,
)

IDENTITY = "5" * 64
EOS = 99


@dataclass(frozen=True)
class _State:
    position: int


def _fresh_proof(position: int, view: int) -> Qwen38MTPQSASelectionProof:
    return Qwen38MTPQSASelectionProof(
        layer_index=0,
        epoch=1,
        source_view_id=view,
        result_view_id=view,
        source_position=position,
        tail_start=0,
        complete_token_count=0,
        complete_indices_key=None,
        valid_token_count=position + 1,
    )


def _seed(position: int = 4, pending: int = 10, d1: int = 11, serial: int = 0) -> Qwen38MTPSeed:
    proof = _fresh_proof(position - 1, 100 + serial * 20)
    return Qwen38MTPSeed(
        position=position,
        current_token_id=pending,
        first_draft_token_id=d1,
        state=_State(position),
        recurrent_residual=object(),
        qsa_selection=proof,
        transaction=object(),
    )


class _Target(Qwen38FixedFiveTarget):
    def __init__(self, target_rows: list[tuple[int, ...]], *, allocated_context: int = MAX_CONTEXT) -> None:
        self._owner = None
        self._poison = None
        self.target_rows = list(target_rows)
        self.verifications: list[Qwen38TargetVerification] = []
        self.commits: list[Qwen38TargetCommit] = []
        self.aborts = 0
        self.released = []
        self.fail_verify = False
        self.fail_commit = False
        self.malformed_commit_roots = False
        self._allocated_context = allocated_context

    @property
    def identity_key(self):
        return IDENTITY

    @property
    def allocated_context(self):
        return self._allocated_context

    @property
    def poisoned(self):
        return self._poison is not None

    @property
    def poisoned_error(self):
        return self._poison

    def claim_runtime_owner(self, owner):
        assert self._owner is None
        self._owner = owner

    def release_runtime_owner(self, owner):
        assert self._owner is owner
        self._owner = None

    def transfer_runtime_owner(self, current_owner, next_owner):
        assert self._owner is current_owner
        self._owner = next_owner

    def state_position(self, state):
        return state.position

    def verify_five(self, input_token_ids, state, *, base_position):
        if self.fail_verify:
            raise RuntimeError("verify fault")
        roots = tuple(object() for _ in range(5))
        verification = Qwen38TargetVerification(
            base_position=base_position,
            input_token_ids=input_token_ids,
            target_token_ids=self.target_rows.pop(0),
            target_hyper_residuals=roots,
            speculative_position=base_position + 5,
            transaction=object(),
        )
        self.verifications.append(verification)
        return verification

    def preflight_commit(self, verification, committed_input_count):
        assert verification is self.verifications[-1]
        assert 1 <= committed_input_count <= 5

    def commit_prefix(self, verification, committed_input_count):
        if self.fail_commit:
            raise RuntimeError("commit fault")
        roots = verification.target_hyper_residuals[:committed_input_count]
        if self.malformed_commit_roots:
            roots = (*roots[:-1], object())
        commit = Qwen38TargetCommit(
            state=_State(verification.base_position + committed_input_count),
            position=verification.base_position + committed_input_count,
            committed_input_token_ids=verification.input_token_ids[:committed_input_count],
            committed_hyper_residuals=roots,
            mode=Qwen38TargetCommitMode.PREFIX_TRANSACTION,
        )
        self.commits.append(commit)
        return commit

    def abort(self, verification):
        self.aborts += 1
        return Qwen38StateRollback(_State(verification.base_position), verification.base_position)

    def release_state(self, state):
        self.released.append(state)


class _MTP(Qwen38MTPDraftEngine):
    def __init__(
        self,
        *,
        next_drafts: list[tuple[int, int, int, int]] | None = None,
        allocated_context: int = MAX_CONTEXT,
    ) -> None:
        self._owner = None
        self._poison = None
        self.next_drafts = list(next_drafts or [(11, 12, 13, 14)])
        self.draft_calls: list[Qwen38MTPDraftBatch] = []
        self.align_calls = []
        self.aborts = 0
        self.released_seeds = []
        self.fail_align = False
        self.bad_chain = False
        self.fail_transfer = False
        self.serial = 1
        self._allocated_context = allocated_context

    @property
    def identity_key(self):
        return IDENTITY

    @property
    def allocated_context(self):
        return self._allocated_context

    @property
    def poisoned(self):
        return self._poison is not None

    @property
    def poisoned_error(self):
        return self._poison

    def claim_runtime_owner(self, owner):
        assert self._owner is None
        self._owner = owner

    def release_runtime_owner(self, owner):
        assert self._owner is owner
        self._owner = None

    def transfer_runtime_owner(self, current_owner, next_owner):
        assert self._owner is current_owner
        if self.fail_transfer:
            raise RuntimeError("MTP owner transfer fault")
        self._owner = next_owner

    def state_position(self, state):
        return state.position

    def bootstrap_shifted_prefill(self, consumed_token_ids, pending_token_id, target_hyper_residuals, state):
        raise NotImplementedError

    def bootstrap_shifted_prefill_rows(self, shifted_rows, state):
        raise NotImplementedError

    def draft_four(self, seed):
        drafts = self.next_drafts.pop(0)
        base = seed.position
        source = seed.qsa_selection.source_view_id
        steps = [
            Qwen38MTPDraftStep(
                step_index=0,
                position=base - 1,
                input_token_id=seed.current_token_id,
                predicted_token_id=drafts[0],
                qsa_selection=seed.qsa_selection,
                reused_qsa_selection=False,
                reused_from_view_id=None,
                precomputed=True,
            )
        ]
        for index in range(1, 4):
            proof = Qwen38MTPQSASelectionProof(
                layer_index=0,
                epoch=seed.qsa_selection.epoch,
                source_view_id=source,
                result_view_id=(source + index if not self.bad_chain or index != 2 else source + 1),
                source_position=base - 1,
                tail_start=seed.qsa_selection.tail_start,
                complete_token_count=seed.qsa_selection.complete_token_count,
                complete_indices_key=seed.qsa_selection.complete_indices_key,
                valid_token_count=seed.qsa_selection.valid_token_count + index,
            )
            steps.append(
                Qwen38MTPDraftStep(
                    step_index=index,
                    position=base + index - 1,
                    input_token_id=drafts[index - 1],
                    predicted_token_id=drafts[index],
                    qsa_selection=proof,
                    reused_qsa_selection=True,
                    reused_from_view_id=source,
                    precomputed=False,
                )
            )
        batch = Qwen38MTPDraftBatch(
            base_position=base,
            seed=seed,
            steps=tuple(steps),
            speculative_position=base + 3,
            transaction=object(),
        )
        self.draft_calls.append(batch)
        return batch

    def preflight_alignment(self, batch, emitted_token_ids, target_hyper_residuals):
        assert len(emitted_token_ids) == len(target_hyper_residuals)

    def commit_alignment(self, batch, emitted_token_ids, target_hyper_residuals):
        if self.fail_align:
            raise RuntimeError("alignment fault")
        base = batch.base_position
        steps = tuple(
            Qwen38MTPAlignmentStep(
                step_index=index,
                position=base + index,
                input_token_id=token,
                qsa_selection=_fresh_proof(base + index, 300 + self.serial * 20 + index),
            )
            for index, token in enumerate(emitted_token_ids)
        )
        self.serial += 1
        state = _State(base + len(emitted_token_ids))
        seed = Qwen38MTPSeed(
            position=state.position,
            current_token_id=emitted_token_ids[-1],
            first_draft_token_id=21 + self.serial,
            state=state,
            recurrent_residual=object(),
            qsa_selection=steps[-1].qsa_selection,
            transaction=object(),
        )
        commit = Qwen38MTPAlignmentCommit(
            state=seed.state,
            position=seed.position,
            aligned_token_ids=emitted_token_ids,
            consumed_hyper_residuals=target_hyper_residuals,
            alignment_steps=steps,
            consumed_seed=batch.seed,
            seed=seed,
        )
        self.align_calls.append(commit)
        return commit

    def advance_seed(self, seed, emitted_token_ids, target_hyper_residuals):
        raise NotImplementedError

    def advance_seed_rows(self, seed, shifted_rows):
        raise NotImplementedError

    def abort(self, batch):
        self.aborts += 1
        return Qwen38StateRollback(batch.seed.state, batch.base_position)

    def release_seed(self, seed):
        self.released_seeds.append(seed)

    def release_state(self, state):
        raise AssertionError("seeded controller must release the seed, not state separately")


def _targets_for_depth(depth: int) -> tuple[int, int, int, int, int]:
    drafts = (11, 12, 13, 14)
    if depth == 4:
        return (*drafts, 77)
    return (*drafts[:depth], 70 + depth, *(80 + index for index in range(4 - depth)))


def test_controller_uses_equal_allocated_capacity_at_c_minus_five_boundary() -> None:
    capacity = 128
    position = capacity - 5
    target = _Target([_targets_for_depth(0)], allocated_context=capacity)
    mtp = _MTP(allocated_context=capacity)
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(position),
        mtp_seed=_seed(position=position),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )

    event = controller.step()

    assert controller.allocated_context == capacity
    assert event.base_position == position
    assert len(target.verifications) == 1


def test_controller_rejects_fixed_five_at_c_minus_four_without_backend_calls(expect_error) -> None:
    capacity = 128
    position = capacity - 4
    target = _Target([_targets_for_depth(0)], allocated_context=capacity)
    mtp = _MTP(allocated_context=capacity)
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(position),
        mtp_seed=_seed(position=position),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )

    with expect_error(Qwen38SpeculativeDecodeUnavailableError, match="allocated context"):
        controller.step()

    assert target.verifications == []
    assert mtp.draft_calls == []


def test_controller_rejects_exhausted_or_mismatched_allocated_capacity(expect_error) -> None:
    capacity = 128
    with expect_error(ValueError, match="exhausted allocated context"):
        Qwen38SpeculativeDecodeController(
            _Target([], allocated_context=capacity),
            _MTP(allocated_context=capacity),
            target_state=_State(capacity),
            mtp_seed=_seed(position=capacity),
            pending_token_id=10,
            eos_token_ids=(EOS,),
        )

    with expect_error(ValueError, match="differs from MTP"):
        Qwen38SpeculativeDecodeController(
            _Target([], allocated_context=capacity),
            _MTP(allocated_context=capacity * 2),
            target_state=_State(4),
            mtp_seed=_seed(position=4),
            pending_token_id=10,
            eos_token_ids=(EOS,),
        )


@pytest.mark.parametrize("depth", range(5))
def test_controller_exact_greedy_depths_shifted_seed_and_root_prefix(depth):
    target = _Target([_targets_for_depth(depth)])
    mtp = _MTP()
    seed = _seed()
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=seed,
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )

    event = controller.step()

    expected_count = depth + 1
    expected_emissions = (*((11, 12, 13, 14)[:depth]), _targets_for_depth(depth)[depth])
    assert tuple(item.token_id for item in event.emissions) == expected_emissions
    assert event.accepted_draft_count == depth
    assert event.committed_input_count == expected_count
    assert event.mtp_extension_count == 3
    assert event.mtp_alignment_count == expected_count
    assert event.next_position == 4 + expected_count
    assert controller.pending_token_id == expected_emissions[-1]
    verification = target.verifications[0]
    target_commit = target.commits[0]
    mtp_commit = mtp.align_calls[0]
    assert target_commit.committed_hyper_residuals == verification.target_hyper_residuals[:expected_count]
    assert target_commit.target_hyper_residual is target_commit.committed_hyper_residuals[-1]
    assert mtp_commit.consumed_hyper_residuals == target_commit.committed_hyper_residuals
    assert mtp_commit.aligned_token_ids == expected_emissions
    assert mtp_commit.consumed_seed is seed


@pytest.mark.parametrize(
    ("targets", "expected", "source"),
    [
        ((EOS, 50, 51, 52, 53), (EOS,), "target_replacement"),
        ((11, EOS, 51, 52, 53), (11, EOS), "target_replacement"),
        ((11, 12, EOS, 52, 53), (11, 12, EOS), "target_replacement"),
        ((11, 12, 13, EOS, 53), (11, 12, 13, EOS), "target_replacement"),
        ((11, 12, 13, 14, EOS), (11, 12, 13, 14, EOS), "target_bonus"),
    ],
)
def test_eos_at_every_commit_depth_still_aligns_authoritative_prefix(targets, expected, source):
    target = _Target([targets])
    mtp = _MTP()
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )
    event = controller.step()
    assert tuple(item.token_id for item in event.emissions) == expected
    assert event.emissions[-1].source.value == source
    assert event.stop_reason is Qwen38SpeculativeStopReason.EOS
    assert controller.status is Qwen38SpeculativeStatus.FINISHED
    assert mtp.align_calls[0].aligned_token_ids == expected


def test_diagnostic_exact_budget_treats_matched_eos_as_an_ordinary_mtp_token() -> None:
    drafts = (11, EOS, 13, 14)
    targets = (*drafts, 77)
    resolution = resolve_greedy_five(
        drafts,
        targets,
        eos_token_ids=(EOS,),
        diagnostic_exact_token_budget=True,
    )
    assert tuple(item.token_id for item in resolution.emissions) == targets
    assert resolution.stop_reason is None

    target = _Target([targets])
    mtp = _MTP(next_drafts=[drafts])
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
        diagnostic_exact_token_budget=True,
    )
    event = controller.step()
    assert tuple(item.token_id for item in event.emissions) == targets
    assert event.stop_reason is None
    assert event.committed_input_count == 5
    assert controller.status is Qwen38SpeculativeStatus.READY
    assert mtp.align_calls[0].aligned_token_ids == targets

    pending_eos_target = _Target([_targets_for_depth(0)])
    pending_eos_mtp = _MTP()
    pending_eos = Qwen38SpeculativeDecodeController(
        pending_eos_target,
        pending_eos_mtp,
        target_state=_State(4),
        mtp_seed=_seed(pending=EOS),
        pending_token_id=EOS,
        eos_token_ids=(EOS,),
        diagnostic_exact_token_budget=True,
    )
    assert pending_eos.status is Qwen38SpeculativeStatus.READY
    pending_eos.close()


def test_diagnostic_exact_budget_requires_an_exact_boolean(expect_error) -> None:
    with expect_error(TypeError, match="diagnostic_exact_token_budget must be boolean"):
        resolve_greedy_five(
            (11, 12, 13, 14),
            _targets_for_depth(4),
            eos_token_ids=(EOS,),
            diagnostic_exact_token_budget=1,
        )


def test_seed_step_is_precomputed_and_only_three_unique_result_views_extend_it():
    target = _Target([_targets_for_depth(0)])
    mtp = _MTP()
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )
    controller.step()
    steps = mtp.draft_calls[0].steps
    assert steps[0].precomputed is True
    assert [step.precomputed for step in steps[1:]] == [False, False, False]
    assert [step.position for step in steps] == [3, 4, 5, 6]
    assert len({step.qsa_selection.result_view_id for step in steps}) == 4
    assert {step.qsa_selection.source_view_id for step in steps} == {steps[0].qsa_selection.source_view_id}
    assert [step.reused_from_view_id for step in steps] == [None, 100, 100, 100]


def test_multiround_uses_alignment_seed_and_handoff_carries_no_target_root():
    target = _Target([_targets_for_depth(0), (23, 40, 41, 42, 43)])
    mtp = _MTP(next_drafts=[(11, 12, 13, 14), (23, 24, 25, 26)])
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )
    first = controller.step()
    second_seed = mtp.align_calls[0].seed
    second = controller.step()
    assert mtp.draft_calls[1].seed is second_seed
    assert second.base_position == first.next_position
    next_target_owner, next_mtp_owner = object(), object()
    handoff = controller.handoff(next_target_owner=next_target_owner, next_mtp_owner=next_mtp_owner)
    assert handoff.mtp_seed is mtp.align_calls[-1].seed
    assert handoff.mtp_state is handoff.mtp_seed.state
    assert not hasattr(handoff, "target_hyper_residual")
    assert controller.status is Qwen38SpeculativeStatus.HANDED_OFF


def test_target_failure_aborts_draft_branch_but_poison_retains_owners(expect_error):
    target = _Target([_targets_for_depth(0)])
    target.fail_verify = True
    mtp = _MTP()
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )
    with expect_error(Qwen38SpeculativeDecodePoisonedError, "verify fault"):
        controller.step()
    assert mtp.aborts == 1
    assert target.aborts == 0
    assert target._owner is not None and mtp._owner is not None
    assert controller.status is Qwen38SpeculativeStatus.POISONED


def test_alignment_failure_after_target_commit_does_not_abort_or_release_roots(expect_error):
    target = _Target([_targets_for_depth(2)])
    mtp = _MTP()
    mtp.fail_align = True
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )
    with expect_error(Qwen38SpeculativeDecodePoisonedError, "alignment fault"):
        controller.step()
    assert len(target.commits) == 1
    assert target.aborts == 0 and mtp.aborts == 0
    assert target.commits[0].committed_hyper_residuals == target.verifications[0].target_hyper_residuals[:3]


def test_malformed_target_root_prefix_is_rejected_after_commit(expect_error):
    target = _Target([_targets_for_depth(1)])
    target.malformed_commit_roots = True
    mtp = _MTP()
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )
    with expect_error(Qwen38SpeculativeDecodePoisonedError, "exact committed target root prefix"):
        controller.step()
    assert not mtp.align_calls


def test_duplicate_qsa_result_view_fails_closed_and_aborts_both_transactions(expect_error):
    target = _Target([_targets_for_depth(0)])
    mtp = _MTP()
    mtp.bad_chain = True
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )
    with expect_error(Qwen38SpeculativeDecodePoisonedError, "result view"):
        controller.step()
    assert mtp.aborts == 1 and target.aborts == 0


def test_close_releases_new_seed_not_stale_roots_or_separate_mtp_state():
    target = _Target([_targets_for_depth(0)])
    mtp = _MTP()
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )
    controller.step()
    live_seed = mtp.align_calls[-1].seed
    controller.close()
    assert mtp.released_seeds == [live_seed]
    assert target._owner is None and mtp._owner is None
    assert controller.status is Qwen38SpeculativeStatus.CLOSED


def test_handoff_rolls_back_first_owner_transfer_if_second_transfer_fails(expect_error):
    target = _Target([_targets_for_depth(0)])
    mtp = _MTP()
    controller = Qwen38SpeculativeDecodeController(
        target,
        mtp,
        target_state=_State(4),
        mtp_seed=_seed(),
        pending_token_id=10,
        eos_token_ids=(EOS,),
    )
    original_target_owner = target._owner
    original_mtp_owner = mtp._owner
    mtp.fail_transfer = True
    with expect_error(Qwen38SpeculativeDecodePoisonedError, "owner transfer fault"):
        controller.handoff(next_target_owner=object(), next_mtp_owner=object())
    assert target._owner is original_target_owner
    assert mtp._owner is original_mtp_owner


def test_constructor_rejects_seed_pending_or_position_mismatch_and_returns_claims(expect_error):
    target = _Target([_targets_for_depth(0)])
    mtp = _MTP()
    with expect_error(ValueError, "seed current token"):
        Qwen38SpeculativeDecodeController(
            target,
            mtp,
            target_state=_State(4),
            mtp_seed=_seed(pending=17),
            pending_token_id=10,
            eos_token_ids=(EOS,),
        )
    assert target._owner is None and mtp._owner is None


def test_pure_greedy_resolution_accepts_zero_through_four():
    for depth in range(5):
        result = resolve_greedy_five((11, 12, 13, 14), _targets_for_depth(depth), eos_token_ids=(EOS,))
        assert result.accepted_draft_count == depth
        assert result.committed_input_count == depth + 1
