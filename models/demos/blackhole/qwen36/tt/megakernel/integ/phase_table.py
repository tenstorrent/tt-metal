"""Megakernel per-bucket phase table (spec §6 interface v0.8.x): one 16-u32 record per phase, in pass order.

Pure python (no ttnn), so the layout is unit-testable on the laptop. The device copy is a uint32 ROW_MAJOR tensor
[1, 1, n_phases, 16] (verify_mk.py). Words 3/4/5/12-15 (weight page base, K/N tiles per core, block-private) are
filled by each block's builder at integration; stage0 leaves them 0.
"""

from dataclasses import dataclass

WORDS = 16

# phase types (spec §6 word 0)
PROJ, MLP_GU, MLP_DN, GDN_STEP, SDPA, AR_SEND, AR_RECV, NORM, STALL, GDN_PREFIX, GDN_STEP_U1 = range(1, 12)
# weight tensor ids (word 2)
W_GATE_UP, W_DOWN, W_GDN_IN, W_GDN_OUT, W_QKV, W_WO, W_KV, W_GDN_STATE = range(8)
# CB ids (spec §6)
CB_X, CB_OUT = 0, 16
# flags (word 11)
F_LAST_IN_LAYER, F_RESIDUAL, F_NORM_FOLDED = 1, 2, 4

NAMES = {
    PROJ: "PROJ", MLP_GU: "MLP_GU", MLP_DN: "MLP_DN", GDN_STEP: "GDN_STEP", SDPA: "SDPA", AR_SEND: "AR_SEND",
    AR_RECV: "AR_RECV", NORM: "NORM", STALL: "STALL", GDN_PREFIX: "GDN_PREFIX", GDN_STEP_U1: "GDN_STEP_U1",
}


@dataclass(frozen=True)
class Phase:
    type: int
    layer: int
    weight: int = 0
    flags: int = 0
    in_cb: int = CB_X
    out_cb: int = CB_OUT

    def record(self, seq):
        w = [0] * WORDS
        w[0], w[1], w[2] = self.type, self.layer, self.weight
        w[6], w[7] = self.in_cb, self.out_cb
        w[8], w[9] = 12, 12  # phase barrier semaphore (program sem 12, cumulative = seq + 1)
        w[11] = self.flags
        w[12] = seq  # block-private word 12: phase sequence number (stage0 probe reads it)
        return w


def layer_phases(layer, full_attention, n_users):
    """Phases of one decoder layer, in pass order (spec §3 chain: in-proj → mixer → out-proj → AR → gate_up → down
    → AR). The AR is one AR_SEND + AR_RECV pair; the receive writes the next CB_X (+ residual, deferred norm)."""
    if full_attention:
        mixer = [Phase(PROJ, layer, W_QKV, F_NORM_FOLDED), Phase(SDPA, layer, W_KV), Phase(PROJ, layer, W_WO)]
    elif n_users == 1:
        mixer = [Phase(PROJ, layer, W_GDN_IN, F_NORM_FOLDED), Phase(GDN_STEP_U1, layer, W_GDN_STATE),
                 Phase(PROJ, layer, W_GDN_OUT)]
    else:
        mixer = [Phase(PROJ, layer, W_GDN_IN, F_NORM_FOLDED), Phase(GDN_PREFIX, layer, W_GDN_STATE),
                 Phase(GDN_STEP, layer, W_GDN_STATE), Phase(PROJ, layer, W_GDN_OUT)]
    ar = [Phase(AR_SEND, layer), Phase(AR_RECV, layer, flags=F_RESIDUAL)]
    mlp = [Phase(MLP_GU, layer, W_GATE_UP, F_NORM_FOLDED), Phase(MLP_DN, layer, W_DOWN)]
    last_ar = [Phase(AR_SEND, layer), Phase(AR_RECV, layer, flags=F_RESIDUAL | F_LAST_IN_LAYER)]
    return mixer + ar + mlp + last_ar


def build_phase_table(full_attention_flags, n_users):
    """Records for one verify bucket: list of n_phases lists of 16 u32, for layer types `full_attention_flags`."""
    assert 1 <= n_users <= 32, n_users
    phases = []
    for i, fa in enumerate(full_attention_flags):
        phases += layer_phases(i, bool(fa), n_users)
    return [p.record(s) for s, p in enumerate(phases)]


def checksum(records):
    """(n_phases, sum of word0, FNV-1a-32 over all words): what the stage0 probe kernel computes on device."""
    s, x = 0, 0x811C9DC5
    for r in records:
        s = (s + r[0]) & 0xFFFFFFFF
        for w in r:
            x = ((x ^ (w & 0xFFFFFFFF)) * 0x01000193) & 0xFFFFFFFF
    return len(records), s, x


def summary(records):
    from collections import Counter

    c = Counter(NAMES[r[0]] for r in records)
    return dict(sorted(c.items()))
