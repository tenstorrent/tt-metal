from ttnn.operations.rms_norm.perf_experiments.scaler_offpath.driver import main

main(
    variants=["prep_first", "after_push", "writer_prep"] * 4, cores=110, c=2, tag="i4_focus_interleaved", keep_log=False
)
