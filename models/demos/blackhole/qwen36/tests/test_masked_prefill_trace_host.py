# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the captured masked-bucket prefill of Qwen36Model (tt/model.py prepare_prefill_traces_masked,
record_prefill_traces_masked, _replay_masked_trace) and the GDN inputs that make that forward trace-safe
(gdn.tp.PrefillMasks, PrefillCheckpoint.buffers, TPGatedDeltaNet.forward_prefill ``masks``).

A masked-bucket forward is the prefill of a prompt that fits one 2048-token chunk, and of the partial last chunk of a
longer one. Eagerly it builds its length-dependent inputs on the host (tokens, RoPE rows, page tables, the GDN masks of
_causal_conv1d_fir and chunk_gated_delta_rule_fused_adapter, the logits row, the checkpoint selectors). The trace reads
them from persistent tensors that each replay restages, and runs the same operations. These tests replace the ttnn
calls with recorders and check: the staged values are the ones the eager forward builds; the traced forward makes no
host write and passes the persistent tensors where the eager one passes host-built ones; the routing of text,
multimodal, checkpointed and resumed rows and of the MTP class's chunk hook; the warmup and release lifecycle.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import model as qwen36_model
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.gdn import fused_chunk
from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp
from models.demos.blackhole.qwen36.tt.gdn.tp import PrefillCheckpoint, PrefillMasks, TPGatedDeltaNet
from models.experimental.gated_attention_gated_deltanet.tt import ttnn_gated_deltanet

KCONV = 4
IMAGE = 9


class Dev:
    """A device tensor stand-in."""

    def __init__(self, name, shape=(), dtype=None):
        self.name, self.shape, self.dtype = name, tuple(shape), dtype

    def __repr__(self):
        return self.name

    def __getitem__(self, key):
        return Dev(f"{self.name}[]", self.shape, self.dtype)


class Host:
    def __init__(self, t, dtype):
        self.t, self.dtype = t, dtype


@pytest.fixture
def device_calls(monkeypatch):
    """Recorders for the ttnn calls of the model-level code: allocations, host-to-device copies, trace calls."""
    calls = []
    count = [0]

    def from_torch(t, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        if device is None:
            return Host(t.clone(), dtype)
        count[0] += 1
        calls.append(("alloc", tuple(t.shape), dtype))
        return Dev(f"buf{count[0]}", t.shape, dtype)

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, dev: calls.append(("stage", dev, host)))
    monkeypatch.setattr(
        ttnn, "execute_trace", lambda device, tid, cq_id=0, blocking=True: calls.append(("replay", tid))
    )
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda device, cq_id=0: calls.append(("begin",)) or len(calls))
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda device, tid, cq_id=0: calls.append(("end", tid)))
    monkeypatch.setattr(ttnn, "release_trace", lambda device, tid: calls.append(("release", tid)))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda device: calls.append(("sync",)))
    monkeypatch.setattr(ttnn, "clone", lambda t, **kw: calls.append(("clone", t)) or Dev(f"copy of {t}"))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: calls.append(("free", t)))
    return calls


class FakeGdn:
    """The GDN layer methods the model calls around a captured prefill."""

    def __init__(self, events, i):
        self.events, self.i = events, i
        self.rec_state, self.conv_carry = Dev(f"rec{i}"), Dev(f"carry{i}")

    def alloc_prefill_trace_buffers(self, checkpoint):
        self.events.append(("gdn alloc", self.i, checkpoint))

    def free_prefill_trace_buffers(self):
        self.events.append(("gdn free", self.i))

    def stage_prefill_init(self, carry):
        self.events.append(("init", self.i, carry))

    def load_prefill_init(self):
        self.events.append(("load init", self.i))

    def note_traced_prefill(self, checkpoint):
        self.events.append(("note", self.i, checkpoint))


def _model(monkeypatch, calls, block_size=1024, buf_blocks=32):
    """A Qwen36Model at TP=8 with 3 GDN layers and 1 attention layer whose forward, logits head, RoPE staging and
    vision splice record their arguments; the chunk trace is prepared and captured."""
    monkeypatch.setattr(qwen36_model, "get_block_size", lambda caches: block_size)
    m = object.__new__(qwen36_model.Qwen36Model)
    m.num_devices = 8
    m.device = m.mesh_device = "mesh"
    m.args = SimpleNamespace(
        gdn_conv_kernel_size=KCONV, hf_config=SimpleNamespace(image_token_id=IMAGE, video_token_id=10)
    )
    m._req_video_grid_thw = None
    gdn = [FakeGdn(calls, i) for i in range(3)]
    m.layers = [SimpleNamespace(is_full_attention=False, attention=g) for g in gdn]
    m.layers.append(SimpleNamespace(is_full_attention=True, attention=None))
    m._paged_kv_caches = "kv"
    m._chunked_chunk_size = 2048
    m._chunked_prepared_key = "prepared"
    m._chunked_trace_id = "chunk trace"
    m._chunked_trace_output = Dev("chunk hidden")
    m._chunk_full_page_table_buf = Dev("chunk full pt", (1, buf_blocks))
    for name in ("token", "page_table", "cos", "sin"):
        setattr(m, f"_chunk_{name}_buf", Dev(f"chunk {name}"))
    m._chunk_start_idx_tensor = Dev("chunk csi")
    m._vis_buf = Dev("vis")
    m._masked_traces = {}
    m._masked_traces_recorded = False
    m._masked_trace_gdn = None
    m._masked_trace_refs = []

    def rope(start, length):
        pos = torch.arange(start, start + length, dtype=torch.float32).reshape(1, 1, length, 1)
        return pos, -pos

    m._rope_tp_cos_sin_torch = rope
    m._build_request_rope = lambda ids, vision: calls.append(("rope", ids.shape[1], vision is not None))
    m._reset_gdn_state_for_new_sequence = lambda: calls.append(("reset",))
    m._set_vision_merge = lambda ids, vision, offset=0: calls.append(
        ("vision", ids.shape[1], vision is not None, offset)
    )

    def forward(token_buf, valid_len, chunk_start, page_table, bucket, inputs=None, checkpoint=False, **kw):
        if inputs is None:
            calls.append(("eager forward", chunk_start, valid_len, bucket, kw.get("checkpoint_len")))
        else:
            calls.append(("traced forward", bucket, checkpoint))
        return Dev(f"hidden {bucket}", (1, 1, bucket, 640), dtype="bf16")

    m._forward_prefill_chunk_masked_tp = forward
    m._forward_prefill_chunk_masked = forward

    def logits(hidden, actual_len, bucket, sel=None):
        calls.append(("logits", bucket, actual_len, sel is not None))
        return Dev(f"logits {bucket}")

    m._masked_bucket_logits_tp = logits
    return m, gdn


def _captured(monkeypatch, calls, **kw):
    m, gdn = _model(monkeypatch, calls, **kw)
    m.prepare_prefill_traces_masked()
    m.record_prefill_traces_masked()
    calls.clear()
    return m, gdn


def _staged(calls):
    """{device buffer: host tensor} of the host-to-device copies in ``calls``."""
    return {c[1]: c[2].t for c in calls if c[0] == "stage"}


# ------------------------------------------------------------------------------- GDN masks equal the eager ones


@pytest.mark.parametrize("valid_len", [1, 2, 3, 100, 255])
def test_prefill_masks_are_the_masks_the_eager_forward_builds(monkeypatch, valid_len):
    """PrefillMasks.host_tensors gives the one-hot that _causal_conv1d_fir builds for the convolution history after
    valid_len tokens (over [K - 1 history rows ; T rows], so a short continuation takes history rows), and the beta / g
    and q / k / v masks of chunk_gated_delta_rule_fused_adapter."""
    T = 256
    built = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: built.append(t) or Dev("host mask"))
    x = Dev("qkv", (1, T, 64))
    ttnn_gated_deltanet._causal_conv1d_fir(
        x, None, None, KCONV, "mesh", conv_state=Dev("carry"), weight_taps=[Dev("w")] * KCONV, valid_len=valid_len
    )
    monkeypatch.setattr(ttnn.transformer, "chunk_gated_delta_rule", lambda *a, **kw: (Dev("o"), Dev("state")))
    q = Dev("q", (1, T, 256), dtype="bf16")
    fused_chunk.chunk_gated_delta_rule_fused_adapter(
        q, q, Dev("v", (1, T, 768)), Dev("beta"), Dev("g"), valid_len=valid_len, qkv_head_dims=(2, 128, 6, 128)
    )
    conv_select, gate, qkv = PrefillMasks.host_tensors(T, valid_len, KCONV)
    assert len(built) == 3
    assert torch.equal(built[0], conv_select) and torch.equal(built[1], gate) and torch.equal(built[2], qkv)


def test_prefill_masks_of_a_whole_bucket_are_ones(expect_error):
    """The eager adapter skips its masks for a whole bucket; the persistent masks are then all ones (a multiply by 1)."""
    T = 128
    conv_select, gate, qkv = PrefillMasks.host_tensors(T, T, KCONV)
    assert gate.eq(1).all() and qkv.eq(1).all()
    assert [int(i) for i in conv_select[0].argmax(-1)] == [T, T + 1, T + 2]
    for bad in (-1, T + 1):
        with expect_error(ValueError, "do not fit"):
            PrefillMasks.host_tensors(T, bad, KCONV)


@pytest.fixture
def op_log(monkeypatch):
    """Record every ttnn operation of the conv and adapter paths as (op, operand names); results are named by the
    count of operations so far, frees are recorded too, host writes are numbered HOST0, HOST1, ..."""
    log = SimpleNamespace(entries=[], ops=0, host=[])

    def op(name):
        def run(*args, **kw):
            log.ops += 1
            log.entries.append((name, tuple(getattr(a, "name", None) for a in args)))
            return Dev(f"{name}{log.ops}", getattr(args[0], "shape", ()), getattr(args[0], "dtype", None))

        return run

    for name in ("concat", "to_layout", "matmul", "multiply", "addcmul", "silu", "reshape", "typecast"):
        monkeypatch.setattr(ttnn, name, op(name))

    def from_torch(t, **kw):
        log.host.append(t)
        return Dev(f"HOST{len(log.host) - 1}")

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.entries.append(("free", (t.name,))))
    monkeypatch.setattr(ttnn.transformer, "chunk_gated_delta_rule", lambda *a, **kw: (Dev("o"), Dev("state")))
    return log


def test_device_masks_run_the_eager_operations_without_a_host_write(op_log):
    """With PrefillMasks' tensors, _causal_conv1d_fir and chunk_gated_delta_rule_fused_adapter run the operations of
    the valid_len path on the same operands, the persistent masks in place of the host-built ones, which they neither
    write nor free."""
    T = 256
    masks = SimpleNamespace(conv_select=Dev("SEL"), gate_mask=Dev("GATE"), qkv_mask=Dev("QKV", dtype="bf16"))

    def run(traced):
        op_log.entries, op_log.ops = [], 0
        ttnn_gated_deltanet._causal_conv1d_fir(
            Dev("qkv", (1, T, 64)),
            None,
            None,
            KCONV,
            "mesh",
            conv_state=Dev("carry"),
            weight_taps=[Dev(f"w{k}") for k in range(KCONV)],
            **({"new_state_select": masks.conv_select} if traced else {"valid_len": 100}),
        )
        fused_chunk.chunk_gated_delta_rule_fused_adapter(
            Dev("q", (1, T, 256), dtype="bf16"),
            Dev("k", (1, T, 256)),
            Dev("v", (1, T, 768)),
            Dev("beta"),
            Dev("g"),
            qkv_head_dims=(2, 128, 6, 128),
            **({"masks": (masks.gate_mask, masks.qkv_mask)} if traced else {"valid_len": 100}),
        )
        return list(op_log.entries)

    eager = run(traced=False)
    assert len(op_log.host) == 3, "the eager path builds the conv selector and the two masks on the host"
    traced = run(traced=True)
    assert len(op_log.host) == 3, "the traced path writes nothing from the host"
    rename = {"HOST0": "SEL", "HOST1": "GATE", "HOST2": "QKV"}
    eager = [(op, tuple(rename.get(o, o) for o in ops)) for op, ops in eager if not (op == "free" and ops[0] in rename)]
    assert traced == eager
    assert ("matmul", ("SEL", "to_layout2")) in traced
    assert [ops[1] for op, ops in traced if op == "multiply" and ops[1] in ("GATE", "QKV")] == ["GATE"] * 2 + [
        "QKV"
    ] * 3


def test_device_masks_need_flat_qkv_of_their_dtype(op_log, expect_error):
    q = Dev("q", (1, 64, 256), dtype="bf16")
    for kw in ({"valid_len": 10}, {}):
        masks = (Dev("GATE"), Dev("QKV", dtype="bf16" if kw else "fp32"))
        with expect_error(ValueError, "device masks replace valid_len"):
            fused_chunk.chunk_gated_delta_rule_fused_adapter(
                q, q, q, Dev("beta"), Dev("g"), qkv_head_dims=(2, 128, 6, 128), masks=masks, **kw
            )


# ------------------------------------------------------------------------------- the GDN layer's traced prefill


def _gdn_layer(monkeypatch, log):
    """A TPGatedDeltaNet at TP=8 dimensions whose projection, conv, delta rule and out projection record their
    arguments; ttnn.copy and ttnn.deallocate are recorded, and a host write raises."""
    dn = object.__new__(TPGatedDeltaNet)
    dn.Nk, dn.Nv, dn.Dk, dn.Dv, dn.K = 2, 6, 128, 128, KCONV
    dn.key_dim_tp, dn.value_dim_tp, dn.qkv_dim_tp = 256, 768, 1280
    dn.tw = {k: Dev(k) for k in ("conv_taps", "neg_exp_A", "dt_bias", "norm_w", "out")}
    dn.mesh, dn.tt_ccl, dn.cfg, dn.scale = "mesh", "ccl", "cfg", 128**-0.5
    dn.args = SimpleNamespace(num_devices=8, ccl_topology=lambda: None)
    dn._gdn_kda_conv, dn._gdn_flat_qkv, dn._gdn_fuse_out = True, True, True
    dn._out_colpar_prefill, dn._fuse_out_mmrs_prefill = False, True
    dn.chunk_seq_masks = dn._fused_const_tiles = dn.gdn_program_config = dn.gdn_wy_inverse = None
    dn._cfg_onehot = "onehot"
    dn.rec_state, dn.conv_carry = Dev("rec_state"), Dev("conv_carry")
    dn.conv_states = [Dev(f"tap{j}") for j in range(KCONV)]
    dn._zero_conv0, dn._zero_rec, dn._zero_conv_carry, dn._zero_rec_f32 = (
        Dev(n) for n in ("z0", "zrec", "zcarry", "zf")
    )
    dn.prefill_checkpoint = None
    dn.prefill_init = None
    dn.prefill_checkpoint_buffers = None
    dn._prefill_init_zero = True
    dn._project_qkvzab = lambda x, T, out_mc=None: (Dev("qkv"), Dev("z"), Dev("a"), Dev("b"))

    def fir(qkv, weight, bias, k, mesh, **kw):
        log.append(("conv", kw.get("valid_len"), getattr(kw.get("new_state_select"), "name", None)))
        return Dev("conv"), Dev("conv_new_state")

    def adapter(q, k, v, beta, g, **kw):
        ck = "masks" not in kw and kw.get("valid_len") is None and beta.name == "ck_beta"
        masks = kw.get("masks")
        log.append(("delta", kw.get("valid_len"), None if masks is None else tuple(t.name for t in masks), ck))
        return (Dev("ck_o"), Dev("ck_rec")) if ck else (Dev("o"), Dev("final_state"))

    def multiply(a, b, **kw):
        shape = getattr(a, "shape", (1, 1, 1, 1))
        if getattr(b, "name", None) == "CK_GATE":
            return Dev("ck_beta" if a.name == "beta" else "ck_g", shape)
        return Dev("product", shape)

    def from_torch(t, device=None, **kw):
        if device is not None:
            raise AssertionError("host write in the GDN prefill")
        return Host(t, kw.get("dtype"))

    monkeypatch.setattr(gdn_tp, "_causal_conv1d_fir", fir)
    monkeypatch.setattr(fused_chunk, "chunk_gated_delta_rule_fused_adapter", adapter)
    monkeypatch.setattr(tpc, "matmul_reduce_scatter_prefill", lambda *a, **kw: Dev("out"))
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape, **kw: Dev(t.name if isinstance(t, Dev) else "view", shape))
    monkeypatch.setattr(ttnn, "sigmoid", lambda t, **kw: Dev("beta"))
    monkeypatch.setattr(ttnn, "multiply", multiply)
    monkeypatch.setattr(ttnn, "matmul", lambda a, b, **kw: Dev("ck_conv" if a.name == "CK_SEL" else "matmul"))
    monkeypatch.setattr(ttnn, "copy", lambda src, dst: log.append(("copy", src.name, dst.name)))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("free", t.name)))
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    return dn


def test_gdn_prefill_with_masks_is_the_masked_forward_without_host_writes(monkeypatch):
    """forward_prefill(masks=...) runs the masked path (the MAC FIR conv, not the KDA conv of a full chunk) with the
    persistent selector and masks where valid_len would build host ones, and writes the workspace as the eager one."""
    log = []
    dn = _gdn_layer(monkeypatch, log)
    T = 256
    x = Dev("x", (1, T, 640))
    masks = SimpleNamespace(T=T, conv_select=Dev("SEL"), gate_mask=Dev("GATE"), qkv_mask=Dev("QKV"))
    dn.forward_prefill(x, valid_len=100, capture_state=True)
    eager = list(log)
    log.clear()
    dn.forward_prefill(x, masks=masks, capture_state=True)
    traced = list(log)
    assert [e for e in eager if e[0] in ("conv", "delta")] == [("conv", 100, None), ("delta", 100, None, False)]
    assert [e for e in traced if e[0] in ("conv", "delta")] == [
        ("conv", None, "SEL"),
        ("delta", None, ("GATE", "QKV"), False),
    ]
    rest = lambda log: [e for e in log if e[0] not in ("conv", "delta")]
    assert rest(eager) == rest(traced), "the same frees and state writes"
    assert ("copy", "final_state", "rec_state") in traced and ("copy", "conv_new_state", "conv_carry") in traced


def test_gdn_traced_checkpoint_goes_into_the_persistent_buffers(monkeypatch):
    """A checkpoint in persistent buffers: the checkpoint call's state and history are copied into the layer's
    checkpoint buffers and freed, and the layer holds the buffers; an eager checkpoint keeps its own tensors."""
    log = []
    dn = _gdn_layer(monkeypatch, log)
    T = 256
    x = Dev("x", (1, T, 640))
    masks = SimpleNamespace(T=T, conv_select=Dev("SEL"), gate_mask=Dev("GATE"), qkv_mask=Dev("QKV"))
    bufs = (Dev("ck_rec_buf"), Dev("ck_conv_buf"))
    dn.prefill_checkpoint_buffers = bufs
    ck = SimpleNamespace(length=None, in_buffers=True, gate_mask=Dev("CK_GATE"), conv_select=Dev("CK_SEL"))
    dn.forward_prefill(x, masks=masks, checkpoint=ck)
    assert ("delta", None, None, True) in log
    assert ("copy", "ck_rec", "ck_rec_buf") in log and ("copy", "ck_conv", "ck_conv_buf") in log
    assert ("free", "ck_rec") in log and ("free", "ck_conv") in log
    assert dn.prefill_checkpoint == bufs
    # Taking and releasing the buffers frees nothing; an eager checkpoint is freed.
    log.clear()
    dn.release_prefill_checkpoint(*dn.take_prefill_checkpoint())
    assert log == []
    eager = SimpleNamespace(length=100, in_buffers=False, gate_mask=Dev("CK_GATE"), conv_select=Dev("CK_SEL"))
    dn.forward_prefill(x, valid_len=200, checkpoint=eager)
    rec, conv = dn.take_prefill_checkpoint()
    assert (rec.name, conv.name) == ("ck_rec", "ck_conv")
    log.clear()
    dn.release_prefill_checkpoint(rec, conv)
    assert log == [("free", "ck_rec"), ("free", "ck_conv")]


def test_gdn_masks_and_buffered_checkpoints_are_refused_outside_their_use(monkeypatch, expect_error):
    log = []
    dn = _gdn_layer(monkeypatch, log)
    x = Dev("x", (1, 256, 640))
    masks = SimpleNamespace(T=256)
    for kw in ({"valid_len": 100}, {"carry_state": False}, {"return_state": True}, {"sp": object()}):
        with expect_error(ValueError, "prefill masks for 256 rows need a carried single-sequence prefill"):
            dn.forward_prefill(x, masks=masks, **kw)
    with expect_error(ValueError, "prefill masks for 128 rows"):
        dn.forward_prefill(x, masks=SimpleNamespace(T=128))
    ck = SimpleNamespace(length=None, in_buffers=True)
    with expect_error(ValueError, "needs alloc_prefill_trace_buffers first"):
        dn.forward_prefill(
            x, masks=SimpleNamespace(T=256, conv_select=None, gate_mask=None, qkv_mask=None), checkpoint=ck
        )


def test_gdn_initial_state_buffers(monkeypatch):
    """stage_prefill_init: a new sequence costs no copies while the initial-state buffers are zero; a continuation
    copies the workspace in, after which the next new sequence copies zeros back once. load_prefill_init (the trace's
    first operations) copies them into the workspace. note_traced_prefill holds the checkpoint buffers."""
    log = []
    dn = _gdn_layer(monkeypatch, log)
    monkeypatch.setattr(ttnn, "from_torch", lambda t, device=None, **kw: Dev(f"zeros{tuple(t.shape)}", t.shape))
    dn.rec_state, dn.conv_carry = Dev("rec_state", (1, 6, 128, 128)), Dev("conv_carry", (1, 3, 1280))
    dn.alloc_prefill_trace_buffers(checkpoint=False)
    init = dn.prefill_init
    dn.alloc_prefill_trace_buffers(checkpoint=True)
    assert dn.prefill_init is init and dn.prefill_checkpoint_buffers is not None
    rec, conv = (t.name for t in init)
    log.clear()
    dn.stage_prefill_init(carry=False)
    assert log == []
    dn.stage_prefill_init(carry=True)
    assert log == [("copy", "rec_state", rec), ("copy", "conv_carry", conv)]
    log.clear()
    dn.stage_prefill_init(carry=False)
    dn.stage_prefill_init(carry=False)
    assert log == [("copy", "zrec", rec), ("copy", "zcarry", conv)]
    log.clear()
    dn.load_prefill_init()
    assert log == [("copy", rec, "rec_state"), ("copy", conv, "conv_carry")]
    dn.note_traced_prefill(checkpoint=True)
    assert dn.prefill_checkpoint == dn.prefill_checkpoint_buffers
    log.clear()
    dn.free_prefill_trace_buffers()
    assert dn.prefill_checkpoint is None and dn.prefill_init is None and dn.prefill_checkpoint_buffers is None
    assert len([e for e in log if e[0] == "free"]) == 4


# ------------------------------------------------------------------------------- warmup, capture and release


@pytest.mark.parametrize("block_size, with_checkpoint", [(1024, [2048]), (512, [1024, 2048]), (2048, [])])
def test_prepare_compiles_every_variant_before_any_capture(monkeypatch, device_calls, block_size, with_checkpoint):
    """prepare_prefill_traces_masked allocates every bucket's inputs (a checkpointed variant only for buckets that a
    KV block boundary falls inside), warms the eager operations around a replay and runs every variant twice; it
    captures nothing. record_prefill_traces_masked captures each variant once; both are idempotent."""
    m, gdn = _model(monkeypatch, device_calls, block_size=block_size)
    m.prepare_prefill_traces_masked()
    assert not any(c[0] in ("begin", "replay") for c in device_calls)
    assert [c for c in device_calls if c[0] == "gdn alloc"] == [
        ("gdn alloc", i, bool(with_checkpoint)) for i in range(3)
    ]
    assert sorted(m._masked_traces) == [128, 256, 512, 1024, 2048]
    assert [b for b, tr in sorted(m._masked_traces.items()) if tr.checkpoint is not None] == with_checkpoint
    for b, tr in m._masked_traces.items():
        assert tr.tok.shape == (1, b) and tr.full_pt.shape == (1, 32) and tr.chunk_pt.shape == (1, -(-b // block_size))
        assert tr.masks.conv_select.shape == (1, KCONV - 1, KCONV - 1 + b) and tr.sel.shape == (1, 1, 1, b)
        assert tr.sel_dtype == "bf16"
    warmed = [c for c in device_calls if c[0] == "traced forward"]
    # one forward that gives the hidden dtype, then two of each variant
    want = [(128, False)] + [
        (b, ck) for b in sorted(m._masked_traces) for ck in m._masked_traces[b].variants() for _ in "xx"
    ]
    assert [(c[1], c[2]) for c in warmed] == want
    staging = [c for c in device_calls if c[0] == "init"]
    assert staging == [(c, i, carry) for i in range(3) for c, carry in [("init", True), ("init", False)]]
    assert ("clone", "logits 128") in [(c[0], getattr(c[1], "name", c[1])) for c in device_calls if c[0] == "clone"]
    assert m._masked_trace_gdn == tuple(t for g in gdn for t in (g.rec_state, g.conv_carry))
    device_calls.clear()
    m.prepare_prefill_traces_masked()
    assert device_calls == []

    m.record_prefill_traces_masked()
    begins = [c for c in device_calls if c[0] == "begin"]
    assert len(begins) == sum(len(tr.variants()) for tr in m._masked_traces.values())
    for tr in m._masked_traces.values():
        assert set(tr.trace_ids) == set(tr.variants()) and set(tr.outputs) == set(tr.variants())
    assert m._masked_traces_recorded and ("reset",) in device_calls
    device_calls.clear()
    m.record_prefill_traces_masked()
    assert device_calls == []


def test_release_returns_masked_prefills_to_the_eager_forward(monkeypatch, device_calls):
    m, gdn = _captured(monkeypatch, device_calls)
    ids = [tid for tr in m._masked_traces.values() for tid in tr.trace_ids.values()]
    m.release_prefill_traces_masked()
    assert sorted(c[1] for c in device_calls if c[0] == "release") == sorted(ids)
    assert [c for c in device_calls if c[0] == "gdn free"] == [("gdn free", i) for i in range(3)]
    assert m._masked_prefill_trace(2048, None) is None
    device_calls.clear()
    m.prefill_masked_bucket(torch.ones(1, 2000, dtype=torch.int64), torch.zeros(1, 4, dtype=torch.int32), 2000)
    assert ("eager forward", 0, 2000, 2048, None) in device_calls and ("reset",) in device_calls


def test_prepare_needs_the_chunk_trace_buffers(monkeypatch, device_calls, expect_error):
    m, _ = _model(monkeypatch, device_calls)
    m._chunked_prepared_key = None
    with expect_error(ValueError, "prepare_prefill_trace_chunked first"):
        m.prepare_prefill_traces_masked()
    with expect_error(ValueError, "call prepare_prefill_traces_masked first"):
        m.record_prefill_traces_masked()


# ------------------------------------------------------------------------------- a replay's inputs


def _replay(m, actual_len, chunk_start=0, checkpoint_len=None, page_table=None, bucket=None):
    bucket = bucket or m._mask_bucket_for(actual_len)
    tokens = torch.arange(1, actual_len + 1, dtype=torch.int32).reshape(1, -1)
    token_buf = torch.cat([tokens, torch.zeros(1, bucket - actual_len, dtype=torch.int32)], dim=1)
    if page_table is None:
        page_table = torch.arange(101, 109, dtype=torch.int32).reshape(1, 8)
    tr = m._masked_traces[bucket]
    return tr, m._replay_masked_trace(tr, token_buf, page_table, actual_len, chunk_start, checkpoint_len)


def test_replay_stages_the_inputs_the_eager_forward_builds(monkeypatch, device_calls):
    """A 2000-token prompt at bucket 2048: the padded tokens, RoPE rows of positions [0, 2048), the chunk start, the
    page table padded to the captured width, the chunk table of its 2 real blocks, the GDN masks of 2000 real rows
    and the logits row 1999. The new sequence starts from zero state; the plain variant replays; the caller gets the
    trace's hidden and a copy of its logits."""
    m, gdn = _captured(monkeypatch, device_calls)
    tr, (hidden, logits) = _replay(m, 2000)
    st = _staged(device_calls)
    assert torch.equal(st[tr.tok][0, :2000], torch.arange(1, 2001, dtype=torch.int32))
    assert st[tr.tok][0, 2000:].eq(0).all()
    assert torch.equal(st[tr.cos].reshape(-1), torch.arange(2048, dtype=torch.float32))
    assert torch.equal(st[tr.sin].reshape(-1), -torch.arange(2048, dtype=torch.float32))
    assert st[tr.csi].tolist() == [0]
    assert st[tr.full_pt].tolist() == [list(range(101, 109)) + [0] * 24]
    assert st[tr.chunk_pt].tolist() == [[101, 102]]
    for t, buf in zip(PrefillMasks.host_tensors(2048, 2000, KCONV), (tr.masks.conv_select, tr.masks.gate_mask)):
        assert torch.equal(st[buf], t)
    assert torch.equal(st[tr.masks.qkv_mask], PrefillMasks.host_tensors(2048, 2000, KCONV)[2])
    assert st[tr.sel].reshape(-1).nonzero().reshape(-1).tolist() == [1999]
    assert tr.checkpoint.gate_mask not in st
    assert [c for c in device_calls if c[0] == "init"] == [("init", i, False) for i in range(3)]
    assert ("replay", tr.trace_ids[False]) in device_calls
    assert [c for c in device_calls if c[0] == "note"] == [("note", i, False) for i in range(3)]
    assert hidden is tr.outputs[False][0] and logits.name == f"copy of {tr.outputs[False][1]}"
    assert not any(c[0] == "sync" for c in device_calls), "a replay overlaps the host work that follows it"


def test_replay_with_a_checkpoint_stages_its_selectors(monkeypatch, device_calls):
    m, _ = _captured(monkeypatch, device_calls)
    tr, _ = _replay(m, 2000, checkpoint_len=1024)
    st = _staged(device_calls)
    gate, select = PrefillCheckpoint.host_tensors(2048, 1024, KCONV)
    assert torch.equal(st[tr.checkpoint.gate_mask], gate) and torch.equal(st[tr.checkpoint.conv_select], select)
    assert ("replay", tr.trace_ids[True]) in device_calls
    assert [c for c in device_calls if c[0] == "note"] == [("note", i, True) for i in range(3)]


def test_replay_of_a_continuation(monkeypatch, device_calls):
    """A row resumed at 2048 (prefix hit or later chunk) with 1716 tokens to go: positions, chunk start and chunk table
    from 2048 on, and the workspace as the initial state."""
    m, _ = _captured(monkeypatch, device_calls)
    tr, _ = _replay(m, 1716, chunk_start=2048, checkpoint_len=1024)
    st = _staged(device_calls)
    assert st[tr.csi].tolist() == [2048]
    assert torch.equal(st[tr.cos].reshape(-1), torch.arange(2048, 4096, dtype=torch.float32))
    assert st[tr.chunk_pt].tolist() == [[103, 104]]
    assert [c for c in device_calls if c[0] == "init"] == [("init", i, True) for i in range(3)]


def test_replay_fills_padding_blocks_into_the_null_block(monkeypatch, device_calls):
    """With 512-token blocks a 1100-token prompt has 3 real blocks of the 4 its bucket can fill: the 4th entry of the
    chunk table is the null block 0 (the eager forward fills 3). A narrow page table pads with zeros, a wide one is
    cut to the captured width."""
    m, _ = _captured(monkeypatch, device_calls, block_size=512)
    tr, _ = _replay(m, 1100)
    st = _staged(device_calls)
    assert st[tr.chunk_pt].tolist() == [[101, 102, 103, 0]]
    device_calls.clear()
    tr, _ = _replay(m, 1100, page_table=torch.arange(1, 41, dtype=torch.int32).reshape(1, 40))
    st = _staged(device_calls)
    assert st[tr.full_pt].tolist() == [list(range(1, 33))] and st[tr.chunk_pt].tolist() == [[1, 2, 3, 0]]


def test_a_replay_waits_for_the_previous_replays_input_copies(monkeypatch, device_calls):
    m, _ = _captured(monkeypatch, device_calls)
    _replay(m, 300)
    assert not any(c[0] == "sync" for c in device_calls)
    device_calls.clear()
    _replay(m, 300)
    first_stage = next(i for i, c in enumerate(device_calls) if c[0] == "stage")
    assert ("sync",) in device_calls[:first_stage]


def test_replay_refusals(monkeypatch, device_calls, expect_error):
    m, _ = _captured(monkeypatch, device_calls)
    with expect_error(ValueError, "needs a prefill of more than 1000"):
        _replay(m, 1000, checkpoint_len=1024, bucket=2048)
    with expect_error(ValueError, "does not hold the prefill's 2 blocks"):
        _replay(m, 2000, page_table=torch.ones(1, 1, dtype=torch.int32))
    with expect_error(ValueError, "starts inside a 1024-token KV block"):
        _replay(m, 100, chunk_start=512)
    assert not any(c[0] in ("init", "stage", "replay") for c in device_calls), "a refused replay touches nothing"


# ------------------------------------------------------------------------------- routing


def test_masked_bucket_prefill_replays_when_a_trace_serves_it(monkeypatch, device_calls):
    """prefill_masked_bucket at start 0: the RoPE table and the vision splice are staged as for the eager forward, the
    GDN reset is the trace's own, the hidden is the trace's output and is not freed by _release_prefill_hidden."""
    m, _ = _captured(monkeypatch, device_calls)
    logits, hidden = m.prefill_masked_bucket(
        torch.ones(1, 600, dtype=torch.int64), torch.zeros(1, 4, dtype=torch.int32), 600, return_hidden=True
    )
    assert ("reset",) not in device_calls
    assert device_calls[:2] == [("rope", 600, False), ("vision", 1024, False, 0)]
    assert ("replay", m._masked_traces[1024].trace_ids[False]) in device_calls
    device_calls.clear()
    m._release_prefill_hidden(hidden)
    m._release_prefill_hidden(Dev("eager hidden"))
    assert [(c[0], c[1].name) for c in device_calls] == [("free", "eager hidden")]


def test_masked_bucket_prefill_stays_eager_without_a_matching_trace(monkeypatch, device_calls):
    """The eager forward runs when no trace serves the call: not captured yet, the SDPA debug path, another GDN
    workspace bound (the batched prefill scratch), or a debug layer capture."""
    m, gdn = _model(monkeypatch, device_calls)
    run = lambda **kw: m.prefill_masked_bucket(
        torch.ones(1, 600, dtype=torch.int64), torch.zeros(1, 4, dtype=torch.int32), 600, **kw
    )
    m.prepare_prefill_traces_masked()
    for case in ("not recorded", "flex", "scratch", "capture"):
        if case == "flex":
            m.record_prefill_traces_masked()
        device_calls.clear()
        if case == "scratch":
            gdn[1].rec_state = Dev("scratch")
        if case == "capture":
            gdn[1].rec_state = m._masked_trace_gdn[2]
            m._capture_layer = 3
        run(**({"flex_sdpa": False} if case == "flex" else {}))
        assert ("reset",) in device_calls and any(c[0] == "eager forward" for c in device_calls), case
        assert not any(c[0] == "replay" for c in device_calls), case


def _chunked(m, end, start=0, checkpoint=0, **kw):
    tokens = torch.full((1, end), 5, dtype=torch.int64)
    if kw.get("vision_tokens") is not None:
        tokens[0, 900:1156] = IMAGE
    return m.prefill_traced_chunked(
        tokens,
        torch.arange(1, 9, dtype=torch.int32).reshape(1, 8),
        actual_len=end,
        start_pos=start,
        checkpoint_pos=checkpoint,
        **kw,
    )


def _replays(m, calls):
    """[(trace, csi)] of the replays in ``calls``: "chunk" or (bucket, checkpoint), with the staged chunk start."""
    names = {tid: (b, ck) for b, tr in m._masked_traces.items() for ck, tid in tr.trace_ids.items()}
    out, csi = [], None
    for c in calls:
        if c[0] == "stage" and c[1] in [tr.csi for tr in m._masked_traces.values()]:
            csi = int(c[2].t[0])
        if c[0] == "replay":
            out.append("chunk" if c[1] == m._chunked_trace_id else (*names[c[1]], csi))
    return out


@pytest.mark.parametrize(
    "case, kw, want",
    [
        ("text up to one chunk", dict(end=2000), [(2048, False, 0)]),
        ("short text", dict(end=100), [(128, False, 0)]),
        ("checkpoint inside the chunk", dict(end=2000, checkpoint=1024), [(2048, True, 0)]),
        ("tail of a long prompt", dict(end=3500), ["chunk", (2048, False, 2048)]),
        ("checkpoint in the tail", dict(end=3500, checkpoint=3072), ["chunk", (2048, True, 2048)]),
        ("whole last chunk with a checkpoint", dict(end=4096, checkpoint=3072), ["chunk", (2048, True, 2048)]),
        ("resumed row", dict(start=2048, end=3764, checkpoint=3072), [(2048, True, 2048)]),
        ("resumed long row", dict(start=1024, end=5000), ["chunk", (2048, False, 3072)]),
    ],
)
def test_every_partial_last_chunk_replays_a_masked_trace(monkeypatch, device_calls, case, kw, want):
    """prefill_traced_chunked of text rows: full chunks replay the chunk trace, the partial (or checkpointed whole)
    last chunk replays the masked trace of its bucket, with the checkpoint variant when the checkpoint lies in it, and
    continues from the workspace when it does not start the sequence."""
    m, _ = _captured(monkeypatch, device_calls)
    _chunked(m, **kw)
    assert _replays(m, device_calls) == want, case
    inits = {c[2] for c in device_calls if c[0] == "init"}
    assert inits == {kw.get("start", 0) > 0 or want[0] == "chunk"}, case
    assert not any(c[0] == "eager forward" for c in device_calls), case


def test_multimodal_rows_splice_their_vision_rows_into_the_replay(monkeypatch, device_calls):
    """A 2000-token multimodal row with an image at [900, 1156) and a checkpoint after 1024 replays the checkpointed
    2048 trace after staging its vision rows; a 3500-token one splices the image into the first chunk's replay and
    stages an empty splice (the vision row offset past the image) for the tail's."""
    m, _ = _captured(monkeypatch, device_calls)
    vision = torch.zeros(256, 16)
    _chunked(m, 2000, checkpoint=1024, vision_tokens=vision)
    first_replay = next(i for i, c in enumerate(device_calls) if c[0] == "replay")
    assert ("vision", 2048, True, 0) in device_calls[:first_replay]
    assert _replays(m, device_calls) == [(2048, True, 0)]
    device_calls.clear()
    _chunked(m, 3500, vision_tokens=vision)
    assert [c for c in device_calls if c[0] == "vision"] == [("vision", 2048, True, 0), ("vision", 2048, True, 256)]
    assert _replays(m, device_calls) == ["chunk", (2048, False, 2048)]


def test_the_mtp_chunk_hook_reads_the_trace_hidden(monkeypatch, device_calls):
    """Qwen36MTPContractForCausalLM's prefill hook (on_chunk) gets the masked trace's hidden for the last chunk, and
    nothing frees it; for a long prompt it gets the chunk trace's output first."""
    m, _ = _captured(monkeypatch, device_calls)
    seen = []
    on_chunk = lambda hidden, start, n: seen.append((hidden, start, n))
    _chunked(m, 2000, checkpoint=1024, on_chunk=on_chunk)
    tr = m._masked_traces[2048]
    assert seen == [(tr.outputs[True][0], 0, 2000)]
    assert not any(c[0] == "free" and c[1] is tr.outputs[True][0] for c in device_calls)
    seen.clear()
    _chunked(m, 3500, on_chunk=on_chunk)
    assert seen == [(m._chunked_trace_output, 0, 2048), (tr.outputs[False][0], 2048, 1452)]


def test_trace_body_reads_only_persistent_inputs(monkeypatch, device_calls):
    """The traced forward passes the trace's tensors to every layer (the device chunk start places the SDPA; the host
    chunk start is 0), the GDN masks and checkpoint buffers in place of valid_len and a host-built checkpoint, makes no
    host write, and frees none of its inputs; the trace body first loads the GDN initial state."""
    m, gdn = _model(monkeypatch, device_calls)
    del m._forward_prefill_chunk_masked_tp
    del m._masked_bucket_logits_tp
    m._vis_buf = None
    m.args.gdn_chunk_size = 64
    m.embd = lambda tok: Dev("embedded", (1, 2048, 640))
    m.norm = lambda x, mode=None: Dev("normed")
    m._lm_head = lambda x: Dev("logits", (1, 1, 1, 8))
    seen = []
    for layer in m.layers:
        layer.forward = lambda x, layer=layer, **kw: seen.append((layer.is_full_attention, kw)) or Dev("x")
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape, **kw: Dev(t.name, shape))
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, mc: t)
    monkeypatch.setattr(ttnn, "matmul", lambda a, b, **kw: Dev(f"{a.name} @ {b.name}"))

    def host_write(*a, **kw):
        raise AssertionError("host write in the traced forward")

    tr = qwen36_model._MaskedPrefillTrace(2048, 2)
    for name in ("tok", "cos", "sin", "csi", "full_pt", "chunk_pt", "sel"):
        setattr(tr, name, Dev(name))
    tr.masks, tr.checkpoint = SimpleNamespace(name="masks"), SimpleNamespace(name="ck buffers")
    monkeypatch.setattr(ttnn, "from_torch", host_write)
    hidden, logits = m._masked_trace_body(tr, True)
    assert [c for c in device_calls if c[0] == "load init"] == [("load init", i) for i in range(3)]
    attn = [kw for full, kw in seen if full]
    gdn_kw = [kw for full, kw in seen if not full]
    assert attn == [
        dict(
            cos=tr.cos,
            sin=tr.sin,
            mode="prefill",
            page_table=tr.full_pt,
            chunk_page_table=tr.chunk_pt,
            chunk_start_idx=0,
            chunk_start_idx_tensor=tr.csi,
            valid_len=None,
        )
    ]
    assert (
        gdn_kw
        == [dict(mode="prefill", chunk_size=64, valid_len=None, gdn_masks=tr.masks, gdn_checkpoint=tr.checkpoint)] * 3
    )
    assert not any(
        c[0] == "free" and c[1] in (tr.tok, tr.cos, tr.sin, tr.csi, tr.full_pt, tr.chunk_pt, tr.sel)
        for c in device_calls
    )
    # Without the checkpoint variant the GDN layers get no checkpoint.
    seen.clear()
    m._masked_trace_body(tr, False)
    assert all(kw["gdn_checkpoint"] is None for full, kw in seen if not full)
