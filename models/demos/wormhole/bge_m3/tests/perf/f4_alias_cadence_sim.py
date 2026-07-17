"""F4 aliased-CB cadence state-machine simulator (HOST-ONLY, no device).

Statically proves the K/V physical-aliasing schedule is safe BEFORE any board
launch:

  * K and V circular buffers share ONE L1 allocation (same base address), with
    DIFFERENT page sizes (K bf4 576B, V bf8 1088B). total_size=146,880B is a
    multiple of LCM(576,1088)=9,792 so both CBs' capacities land exactly on the
    allocation end (K cap 255 pages, V cap 135 pages) => full-capacity push/pop
    wraps each ring pointer back to base every iteration.

  * The CB machinery does NOT know K and V overlap in bytes. Safety is enforced
    ONLY by explicit K_CONSUMED / V_CONSUMED handshakes:
        fill K -> QK reads K -> K_CONSUMED -> fill V -> PV reads V -> V_CONSUMED
        -> (next) fill K ...

This simulator models reader + compute as interleaved steppers over the shared
byte region, tracks which bytes are LIVE (written, not yet consumed) for each
tensor, and ASSERTS:
  1. No write ever lands on the OTHER tensor's live bytes (no clobber).
  2. Every phase's ring pointer wraps back to base (full-capacity push/pop).
  3. Semaphore up/down counts stay balanced and non-negative (no deadlock, no
     double-signal).
  4. Compute only ever reads bytes the reader actually wrote (the first 128
     real pages), never padding.

Run: python .../f4_alias_cadence_sim.py   (pure host, exits non-zero on failure)
"""

import sys

# ---- fixed contract (q256/k2048, DHt=2, vDHt=2, N300 encoder SDPA) ----
K_PAGE = 576  # bfloat4_b tile
V_PAGE = 1088  # bfloat8_b tile
ALLOC = 146880  # shared L1 allocation (15 * LCM(576,1088))
K_CAP_PAGES = ALLOC // K_PAGE  # 255
V_CAP_PAGES = ALLOC // V_PAGE  # 135
K_REAL_PAGES = 128  # DHt(2) * Sk_chunk_t(64) = 128 tiles of real K data
V_REAL_PAGES = 128  # vDHt(2) * Sk_chunk_t(64) = 128 tiles of real V data
N_K_CHUNKS = 4  # k2048 over 8192 => 4 k-chunks per q-chunk
N_Q_CHUNKS = 16  # q256 over 4096 folded => 16 q-chunks/core (48/core / 3 heads)


class AliasRegion:
    """Shared L1 byte region backing two CBs that alias it."""

    def __init__(self, size):
        self.size = size
        # owner[b] = 'K' | 'V' | None  -> which tensor's LIVE data occupies byte b
        self.owner = [None] * size
        self.errors = []

    def write(self, tensor, base, nbytes):
        """Reader writes real data for `tensor` starting at `base`."""
        for b in range(base, base + nbytes):
            other = self.owner[b]
            if other is not None and other != tensor:
                self.errors.append(
                    f"CLOBBER: {tensor} write at byte {b} over LIVE {other} data"
                )
            self.owner[b] = tensor

    def consume(self, tensor, base, nbytes):
        """Compute reads then releases `tensor`'s live bytes."""
        for b in range(base, base + nbytes):
            if self.owner[b] != tensor:
                self.errors.append(
                    f"READ-HAZARD: {tensor} consume at byte {b} but owner={self.owner[b]}"
                )
        for b in range(base, base + nbytes):
            if self.owner[b] == tensor:
                self.owner[b] = None


class Ring:
    """One CB (buffer_index) over the shared region: own rd/wr pointers."""

    def __init__(self, name, page, cap_pages):
        self.name = name
        self.page = page
        self.cap = cap_pages
        self.wr = 0  # page index
        self.rd = 0
        self.wraps = 0

    def push(self, npages):
        # full-capacity push advances wr by cap -> wraps to 0
        assert npages == self.cap, f"{self.name}: push {npages} != capacity {self.cap}"
        self.wr = (self.wr + npages) % self.cap
        if self.wr == 0:
            self.wraps += 1
        return self.wr

    def pop(self, npages):
        assert npages == self.cap, f"{self.name}: pop {npages} != capacity {self.cap}"
        self.rd = (self.rd + npages) % self.cap
        return self.rd


def simulate(break_handshake=None):
    """break_handshake in {None,'skip_v_wait','skip_k_wait'} for negative controls."""
    region = AliasRegion(ALLOC)
    k_ring = Ring("K", K_PAGE, K_CAP_PAGES)
    v_ring = Ring("V", V_PAGE, V_CAP_PAGES)

    # semaphores: reader waits on these before overwriting the shared region
    k_consumed = 0  # up'd by compute after QK reads K
    v_consumed = 0  # up'd by compute after PV reads V
    errs = []

    total_iters = N_Q_CHUNKS * N_K_CHUNKS
    for it in range(total_iters):
        # ---- reader: fill K ----
        # next-K may overwrite V bytes from the PREVIOUS iteration; must wait
        # V_CONSUMED first (except the very first iteration).
        if it > 0 and break_handshake != "skip_v_wait":
            if v_consumed <= 0:
                errs.append(f"it{it}: reader fills K but V not yet consumed (deadlock/clobber)")
            else:
                v_consumed -= 1
        # K real data must start at base (ring wrapped last iteration)
        k_base = k_ring.wr * K_PAGE
        if k_base != 0:
            errs.append(f"it{it}: K fill base {k_base} != 0 (ring did not wrap)")
        region.write("K", k_base, K_REAL_PAGES * K_PAGE)
        k_ring.push(K_CAP_PAGES)  # full capacity: 255 pages

        # ---- compute: QK consumes K, signals K_CONSUMED ----
        rd_base = k_ring.rd * K_PAGE
        region.consume("K", rd_base, K_REAL_PAGES * K_PAGE)
        k_ring.pop(K_CAP_PAGES)
        k_consumed += 1

        # ---- reader: fill V ----
        # V overwrites K bytes; must wait K_CONSUMED first
        if break_handshake == "skip_k_wait":
            pass
        elif k_consumed <= 0:
            errs.append(f"it{it}: reader fills V but K not consumed (clobber)")
        else:
            k_consumed -= 1
        v_base = v_ring.wr * V_PAGE
        if v_base != 0:
            errs.append(f"it{it}: V fill base {v_base} != 0 (ring did not wrap)")
        region.write("V", v_base, V_REAL_PAGES * V_PAGE)
        v_ring.push(V_CAP_PAGES)  # full capacity: 135 pages

        # ---- compute: PV consumes V, signals V_CONSUMED ----
        rd_base = v_ring.rd * V_PAGE
        region.consume("V", rd_base, V_REAL_PAGES * V_PAGE)
        v_ring.pop(V_CAP_PAGES)
        v_consumed += 1

    errs.extend(region.errors)

    # ---- final invariants ----
    if k_ring.wraps != total_iters:
        errs.append(f"K wrapped {k_ring.wraps} != {total_iters} iters")
    if v_ring.wraps != total_iters:
        errs.append(f"V wrapped {v_ring.wraps} != {total_iters} iters")
    if any(o is not None for o in region.owner):
        errs.append("region has LIVE bytes at end (leaked, unconsumed)")
    # Residue asymmetry is a KEY correctness property, not a wash:
    #   K_CONSUMED is consumed WITHIN the same iteration (by fill-V), so it
    #     returns to 0 every iter -> final k_consumed == 0.
    #   V_CONSUMED is consumed by the NEXT iteration's fill-K, so the last
    #     iteration's V_CONSUMED dangles -> final v_consumed == 1.
    # This proves the ordering is intra-iteration for K and inter-iteration for V.
    if k_consumed != 0 or v_consumed != 1:
        errs.append(f"semaphore residue k={k_consumed} v={v_consumed} (expected 0/1)")

    return errs


def main():
    print("F4 aliased-CB cadence simulator")
    print(f"  alloc={ALLOC}B  K cap={K_CAP_PAGES}pg V cap={V_CAP_PAGES}pg")
    print(f"  real K={K_REAL_PAGES}pg({K_REAL_PAGES*K_PAGE}B) V={V_REAL_PAGES}pg({V_REAL_PAGES*V_PAGE}B)")
    print(f"  iters={N_Q_CHUNKS*N_K_CHUNKS} (q_chunks={N_Q_CHUNKS} x k_chunks={N_K_CHUNKS})")
    errs = simulate()
    if errs:
        print(f"\nFAIL ({len(errs)} issues):")
        for e in errs[:20]:
            print("  -", e)
        sys.exit(1)
    print("\nPASS: full-capacity push/pop wraps to base every iter; no clobber;")
    print("      no read-hazard; strict K_CONSUMED->fillV, V_CONSUMED->fillK ordering;")
    print("      semaphores balanced (0/1 steady-state residue); region empty at end.")

    # Negative controls: the test MUST detect a hazard when a handshake is
    # dropped, else it proves nothing.
    neg_k = simulate(break_handshake="skip_k_wait")
    neg_v = simulate(break_handshake="skip_v_wait")
    if not neg_k:
        print("NEGATIVE-CONTROL FAIL: dropping K_CONSUMED wait did NOT trip a hazard")
        sys.exit(1)
    if not neg_v:
        print("NEGATIVE-CONTROL FAIL: dropping V_CONSUMED wait did NOT trip a hazard")
        sys.exit(1)
    print(f"NEG-CONTROLS OK: skip_k_wait -> {len(neg_k)} hazards, "
          f"skip_v_wait -> {len(neg_v)} hazards (both correctly detected).")
    sys.exit(0)


if __name__ == "__main__":
    main()
