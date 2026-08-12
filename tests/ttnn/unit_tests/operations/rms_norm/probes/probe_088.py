from ttnn.operations.rms_norm.perf_experiments.scaler_offpath.driver import main

main(variants=["prep_first"], cores=8, c=3, mask=True, tag="i4_masksmoke", keep_log=False)
