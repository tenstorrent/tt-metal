# Round 3 eltwise binary: seed torch, numpy and random before every test, so that tests drawing from the global generators
# without a seed get the same inputs in two processes.
import random

import numpy as np
import torch


def pytest_runtest_setup(item):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
