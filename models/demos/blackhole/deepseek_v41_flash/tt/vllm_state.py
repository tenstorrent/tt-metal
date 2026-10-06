# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free bookkeeping of the vLLM interface of DeepSeek-V4.1-Flash (``tt/generator_vllm.py``): the slot map, the per-slot token history and the
batch building / padding / output scatter helpers. Only ``torch`` is imported here, so ``tests/test_vllm_interface.py`` runs without a device.

Vocabulary
  * *logical slot*: the row of the vLLM TT plugin's persistent batch / state-slot table (0 .. max_num_seqs-1). The plugin hands a new request a free logical slot
    (``empty_slots`` of ``prefill_forward``) and permutes the slots of the running requests between decode steps (``slot_remap`` of ``decode_forward``: row i of the
    new layout reads old slot ``slot_remap[i]``).
  * *physical slot*: the model's user index b (0 .. B-1, B = 4 mesh rows x U users per row; user b lives on mesh row b // U with its pages, rings, compressor state,
    index keys and Engram history). Device state never moves: the permutation is applied to this host table instead (``SlotTable.apply_remap``).
"""

import torch

MESH_ROWS = 4
MAX_USERS_PER_ROW = 32  # mHC kernels: at most 32 users per mesh row
PAGE_TOKENS = 128


def padded_batch(max_num_seqs: int, rows: int = MESH_ROWS) -> int:
    """Model batch for ``max_num_seqs`` requests: a multiple of the mesh rows (users per row = ceil(n / rows)), at most 32 users per row."""
    if max_num_seqs < 1:
        raise ValueError(f"max_num_seqs must be >= 1, got {max_num_seqs}")
    b = -(-max_num_seqs // rows) * rows
    if b // rows > MAX_USERS_PER_ROW:
        raise ValueError(
            f"max_num_seqs {max_num_seqs} needs {b // rows} users per mesh row; at most {MAX_USERS_PER_ROW} (batch 128)"
        )
    return b


def s_pad_bucket(max_len: int, chunk: int | None, policy: str = "bucket") -> int:
    """Padded prompt length (``s_pad_max`` of ``Model.prefill_forward``) of one prefill call. The captured prefill trace is keyed by (chunk, S_pad) and a longer
    S_pad than the one in the process is a teardown + recapture (and can OOM), so S_pad is rounded up to a power-of-two multiple of 1024 tokens (policy
    ``bucket``), is exact (``exact``: ceil to the chunk, the demo's behaviour) or a fixed integer (``DSV41_VLLM_S_PAD=<tokens>``: every prompt is padded to it).
    """
    unit = chunk or PAGE_TOKENS
    if policy == "exact":
        return -(-max_len // unit) * unit
    if policy.isdigit():
        fixed = int(policy)
        if max_len > fixed:
            raise ValueError(f"prompt of {max_len} tokens exceeds DSV41_VLLM_S_PAD={fixed}")
        return -(-fixed // unit) * unit
    if policy != "bucket":
        raise ValueError(f"unknown S_pad policy {policy!r} (bucket | exact | <tokens>)")
    b = 1024
    while b < max_len:
        b *= 2
    return -(-b // unit) * unit


class SlotTable:
    """logical slot -> physical slot map (an injective map of ``n_logical`` plugin slots into ``n_phys`` model users) and the set of occupied physical slots."""

    def __init__(self, n_logical: int, n_phys: int):
        if n_logical > n_phys:
            raise ValueError(f"{n_logical} plugin slots exceed the model batch {n_phys}")
        self.n_logical, self.n_phys = n_logical, n_phys
        self.phys = list(range(n_logical))
        self.live: set[int] = set()  # occupied PHYSICAL slots

    def physical(self, logical: int) -> int:
        return self.phys[logical]

    def claim(self, logical_slots) -> list[int]:
        """Prefill: the physical slots of the (free) logical slots the plugin assigned to new requests. A slot that is still occupied is a re-prefill of the same
        request (preemption / restart) and is claimed again."""
        out = []
        for s in logical_slots:
            if not 0 <= int(s) < self.n_logical:
                raise ValueError(f"empty slot {s} outside 0..{self.n_logical - 1}")
            out.append(self.phys[int(s)])
        if len(set(out)) != len(out):
            raise ValueError(f"duplicate prefill slots {list(logical_slots)}")
        self.live.update(out)
        return out

    def bind(self, logical: int, phys: int) -> None:
        """A chunk continuation arrives in logical slot ``logical`` (the plugin re-picks the state slot of a partly prefilled request every step) while its model user is
        ``phys``: make the slot map point at ``phys`` (swap with the logical slot that currently maps to it, so the map stays a permutation).
        """
        cur = self.phys[logical]
        if cur == phys:
            return
        j = self.phys.index(phys)
        self.phys[j], self.phys[logical] = cur, phys

    def release(self, logical: int) -> int | None:
        """The request in logical slot ``logical`` finished / was preempted: free its physical slot. Returns it (None when the slot held no request)."""
        if not 0 <= logical < self.n_logical:
            return None
        p = self.phys[logical]
        if p not in self.live:
            return None
        self.live.discard(p)
        return p

    def apply_remap(self, remap) -> None:
        """Decode: new logical slot i holds what old logical slot ``remap[i]`` held. ``remap`` must be a permutation of 0..n_logical-1 (the plugin's contract)."""
        remap = [int(x) for x in (remap.tolist() if hasattr(remap, "tolist") else remap)]
        if sorted(remap) != list(range(self.n_logical)):
            raise ValueError(f"slot_remap is not a permutation of 0..{self.n_logical - 1}: {remap}")
        self.phys = [self.phys[r] for r in remap]


class TokenBook:
    """Token history per PHYSICAL slot: the prompt plus every token fed to the model. Needed to re-prefill a live user (``context``), nothing else reads it."""

    def __init__(self, n_phys: int, capacity: int):
        self.tok = torch.zeros(n_phys, capacity, dtype=torch.int32)
        self.n = torch.zeros(n_phys, dtype=torch.long)  # tokens whose KV the model holds

    def set_prompt(self, p: int, tokens: torch.Tensor) -> None:
        n = int(tokens.numel())
        if n > self.tok.shape[1]:
            raise ValueError(f"prompt of {n} tokens exceeds the history capacity {self.tok.shape[1]}")
        self.tok[p, :n] = tokens.to(torch.int32)
        self.n[p] = n

    def note_fed(self, p: int, pos: int, token: int) -> None:
        """The decode step fed ``token`` at position ``pos`` of slot p."""
        if pos >= self.tok.shape[1]:
            raise ValueError(f"position {pos} exceeds the history capacity {self.tok.shape[1]}")
        self.tok[p, pos] = int(token)
        self.n[p] = pos + 1

    def context(self, p: int) -> torch.Tensor:
        return self.tok[p, : int(self.n[p])].long()

    def clear(self, p: int) -> None:
        self.n[p] = 0


class PrefillTracker:
    """Users whose prompt is being prefilled in chunks (or just finished and not yet decoding), by physical slot: {phys: (end position, prompt tokens [:end])}.
    The plugin gives a chunk continuation no stable id (its state slot is re-picked every step), so a continuation is recognised by its token prefix. While such a user
    is not part of a decode step the decode loop feeds it the PARKED position ``end`` (a position nothing of its prompt state lives at) instead of token 0 at position 0
    (position 0 would overwrite its ring row 0, its first latent and its Engram history of the first token)."""

    def __init__(self):
        self.users: dict[int, tuple[int, torch.Tensor]] = {}

    def update(self, p: int, end: int, tokens: torch.Tensor) -> None:
        self.users[p] = (int(end), tokens[: int(end)].clone().reshape(-1).long())

    def find(self, start: int, prefix: torch.Tensor):
        """physical slot of the in-progress user whose processed prefix is exactly ``prefix`` (``start`` tokens), else None."""
        prefix = prefix.reshape(-1).long()
        for p, (end, toks) in self.users.items():
            if end == start and toks.numel() == start and torch.equal(toks, prefix):
                return p
        return None

    def drop(self, p: int) -> None:
        self.users.pop(p, None)

    def parked(self) -> dict[int, int]:
        return {p: end for p, (end, _) in self.users.items()}


def build_prefill_batch(B, phys_new, tokens, prompt_lens, live_context=None):
    """-> (tokens_B [B, L] int64, lens_B [B] int64, active_B [B] bool, index_of {physical slot: row in ``tokens``}).

    ``tokens`` [N, L] right padded prompts of the new requests, ``phys_new`` their physical slots, ``prompt_lens`` their lengths. ``live_context`` {physical slot:
    1-D token tensor}: users that are re-prefilled from their history instead of being left alone (``active`` stays True for them too). Rows of users that are not
    prefilled hold token 0 and length 0."""
    N = tokens.shape[0]
    lens = [int(x) for x in (prompt_lens.tolist() if hasattr(prompt_lens, "tolist") else prompt_lens)]
    if len(lens) != N or len(phys_new) != N:
        raise ValueError(f"{N} prompts, {len(lens)} lengths, {len(phys_new)} slots")
    ctx = live_context or {}
    L = max([max(lens, default=0)] + [int(c.numel()) for c in ctx.values()])
    toks = torch.zeros(B, L, dtype=torch.long)
    lens_b = torch.zeros(B, dtype=torch.long)
    act = torch.zeros(B, dtype=torch.bool)
    index_of = {}
    for i, p in enumerate(phys_new):
        if lens[i] < 1 or lens[i] > tokens.shape[1]:
            raise ValueError(f"prompt length {lens[i]} outside 1..{tokens.shape[1]}")
        toks[p, : lens[i]] = tokens[i, : lens[i]].long()
        lens_b[p], act[p] = lens[i], True
        index_of[p] = i
    for p, c in ctx.items():
        if p in index_of:
            continue
        toks[p, : c.numel()] = c.long()
        lens_b[p], act[p] = int(c.numel()), True
    return toks, lens_b, act, index_of


def build_decode_inputs(B, slots: SlotTable, tokens, start_pos, parked=None):
    """Plugin decode inputs (row i = logical slot i; ``start_pos`` -1 = padding row) -> model inputs.
    -> (tok_B [B] int64, pos_B [B] int64, rows [(logical row, physical slot, position)]). Unused physical slots feed token 0 at position 0, except the PARKED ones
    ({physical slot: position}, users that are being prefilled in chunks and are not decoding yet): token 0 at their parked position.
    """
    tok = tokens.reshape(-1).long()
    pos = start_pos.reshape(-1).long()
    if tok.numel() != pos.numel():
        raise ValueError(f"{tok.numel()} tokens vs {pos.numel()} positions")
    if tok.numel() > slots.n_logical:
        raise ValueError(f"decode batch of {tok.numel()} rows exceeds the {slots.n_logical} plugin slots")
    tok_b = torch.zeros(B, dtype=torch.long)
    pos_b = torch.zeros(B, dtype=torch.long)
    rows = []
    for i in range(tok.numel()):
        if int(pos[i]) < 0:
            continue
        p = slots.physical(i)
        tok_b[p], pos_b[p] = tok[i], pos[i]
        rows.append((i, p, int(pos[i])))
    taken = {p for _, p, _ in rows}
    for p, pp in (parked or {}).items():
        if p not in taken:
            pos_b[p] = int(pp)
    return tok_b, pos_b, rows


def scatter_rows(values, rows, width, fill=0):
    """values [B, ...] indexed by physical slot -> [width, ...] in plugin row order (padding rows = ``fill``)."""
    out = torch.full((width, *values.shape[1:]), fill, dtype=values.dtype)
    for i, p, _ in rows:
        out[i] = values[p]
    return out


def _col(x, rows):
    if x is None:
        return None
    v = x.reshape(-1).tolist() if isinstance(x, torch.Tensor) else (list(x) if isinstance(x, (list, tuple)) else [x])
    return v if rows is None else [v[i] for i in rows if i < len(v)]


def sampling_wants_greedy(sampling_params, rows=None) -> bool:
    """True when every (selected) row of the device-sampling params is greedy (temperature 0, or top_k 1). The model samples greedily on the device only."""
    t, k = _col(getattr(sampling_params, "temperature", None), rows), _col(
        getattr(sampling_params, "top_k", None), rows
    )
    if t is None:
        return True
    k = k if k is not None else [0] * len(t)
    return all(float(ti) == 0.0 or int(ki) == 1 for ti, ki in zip(t, k))


def wants_logprobs(sampling_params, rows=None) -> bool:
    e = _col(getattr(sampling_params, "enable_log_probs", None), rows)
    return bool(e) and any(bool(x) for x in e)
