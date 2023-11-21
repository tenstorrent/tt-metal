from typing import List
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_op import TTPyOp


class TTPyUntilizeWithHalo(TTPyOp):
    # cache map for kernel configs corresponding to unique sliding window op params
    # sliding window op params: tuple(stride_hw: tuple(int, int), pad_hw: tuple(int, int), filter_hw: tuple(int, int), input_nhw: tuple(int, int, int), num_cores_nhw: int)
    static_kernel_configs_cache_map = {}

    def __init__(self, sliding_window_op_params):
        self.sliding_window_op_params = sliding_window_op_params

    # override abstract methods from base class TTPyOp
    def set_op_configs(self):
        return
