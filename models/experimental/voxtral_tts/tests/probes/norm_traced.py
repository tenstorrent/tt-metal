"""Re-test the SHARDED rms_norm inside the trace. 6.39/6.40 rejected it at +4.4 ms/step eagerly.

That rejection was an op-count argument -- "the reshard is the cost" -- and 52 reshards at the
eager ~68 us launch floor is ~3.5 ms of the 4.4. 6.65 traced the frame, so a reshard now costs
~2.6 us (measured: that is what concat dropped to). If the sharded norm's own kernel is faster
than the interleaved one's 63.5 us, the arithmetic flips.

Legality is unchanged and is a property of the tensor (6.39): 3072 wide is 96 tiles, a tile is
indivisible, so cores * block_w must be 96.

Measured on Block 2's traced _solve, which contains 49 norm calls per frame.
"""
import json, os, time
import torch, ttnn
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    FM_NORM_EPS, N_ACOUSTIC_CODEBOOK)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline
HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
REPS, ROUNDS, NT = 30, 5, 96          # 3072 / 32 = 96 tiles wide
dev = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=250*1024*1024)
try:
    pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
    fl = pipe.flow
    case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][2]
    e = pref.build_inputs_embeds(torch.tensor(case["ids"], dtype=torch.long),
                                 pref.load_voice(case["voice"]), pipe.wb)
    h = pipe.backbone.prefill_last(e)[:, 0]
    torch.manual_seed(0)
    xd = fl._up(torch.randn(1, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32)
    hd = fl._up(fl._cfg_input(1, h))
    _norm0 = flow.TtVoxtralFlow._norm

    def make(gx, gy):
        nc = gx * gy
        assert NT % nc == 0, f"{nc} cores does not divide {NT} tiles"
        # use_height_and_width_as_shard_shape: without it the tuple is read as a TENSOR shape,
        # which is what made every config fail with "Physical shard shape ... must be tile sized".
        shard = ttnn.create_sharded_memory_config(
            (32, 3072 // nc), core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)
        prg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, gy), subblock_w=1, block_h=1,
            block_w=NT // nc, inplace=False)
        def _norm(self, x, g):
            xs = ttnn.to_memory_config(x, shard)
            r = ttnn.rms_norm(xs, weight=g, epsilon=FM_NORM_EPS, program_config=prg,
                              memory_config=shard, compute_kernel_config=flow.COMPUTE_CONFIG)
            return ttnn.to_memory_config(r, flow._L1)
        return _norm

    def timed(fn):
        flow.TtVoxtralFlow._norm = fn
        fl._solve(xd, hd, 1, flow.N_DECODING_STEPS, flow.CFG_ALPHA)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try: out = fl._solve(xd, hd, 1, flow.N_DECODING_STEPS, flow.CFG_ALPHA)
        finally: ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        r = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(REPS): ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            r.append((time.perf_counter() - t0) / REPS * 1e3)
        val = ttnn.to_torch(out).float().clone()
        ttnn.release_trace(dev, tid)
        return sum(r) / len(r), val

    print(f"  {'norm config':<28} {'cores':>6} {'block_w':>8} {'ms/frame':>10} {'vs ships':>9} "
          f"{'max|delta| vs ships':>20}")
    base, ref = timed(_norm0)
    print(f"  {'interleaved (ships)':<28} {'-':>6} {'-':>8} {base:>10.3f} {0.0:>+9.3f} {'-':>20}")
    for gx, gy in ((12, 8), (8, 6), (8, 4), (8, 3), (8, 2)):
        nc = gx * gy
        if NT % nc: continue
        try:
            ms, val = timed(make(gx, gy))
            d = float((val - ref).abs().max())
            print(f"  {'sharded ' + f'{gx}x{gy}':<28} {nc:>6} {NT//nc:>8} {ms:>10.3f} "
                  f"{base-ms:>+9.3f} {d:>20.3e}")
        except Exception as ex:
            print(f"  {'sharded ' + f'{gx}x{gy}':<28} {nc:>6} {NT//nc:>8} "
                  f"FAILED: {type(ex).__name__}: {str(ex).splitlines()[0][:38]}")
    print("\n  6.39 eager: sharded 8x4 was +4.381 ms/step WORSE than interleaved.")
finally:
    flow.TtVoxtralFlow._norm = _norm0
    ttnn.close_device(dev)
