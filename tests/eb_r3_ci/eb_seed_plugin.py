# Round 3 eltwise binary: seed torch, numpy and random before every test, so that tests drawing from the global generators
# without a seed get the same inputs in two processes.
import os as _os_guard, sys as _sys_guard
if not (_os_guard.environ.get("HWLOCK_HELD") or _os_guard.environ.get("GITHUB_ACTIONS")):
    _sys_guard.exit("not under hwlock")
import random

import numpy as np
import torch


def pytest_runtest_setup(item):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
