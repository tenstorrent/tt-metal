import os, sys, traceback


class Tracer(dict):
    def __setitem__(self, k, v):
        if k == "RMS_STAGE_ZONES":
            print(f"SET RMS_STAGE_ZONES={v!r}", flush=True)
            traceback.print_stack()
        super().__setitem__(k, v)


os.environ._data = os.environ._data  # noqa
_real_set = type(os.environ).__setitem__


def hook(self, k, v):
    if k == "RMS_STAGE_ZONES":
        print(f"SET RMS_STAGE_ZONES={v!r}", flush=True)
        traceback.print_stack()
    _real_set(self, k, v)


type(os.environ).__setitem__ = hook
print("env at start:", os.environ.get("RMS_STAGE_ZONES"), flush=True)
sys.path.insert(
    0,
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_mcast_v2",
)
import bench_v2

print("after bench_v2:", os.environ.get("RMS_STAGE_ZONES"), flush=True)
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD

print("PD.STAGE_ZONES:", PD.STAGE_ZONES, "env:", os.environ.get("RMS_STAGE_ZONES"), flush=True)
