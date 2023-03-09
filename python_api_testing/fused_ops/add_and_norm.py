from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

from gpai import gpai
from python_api_testing.fused_ops.layernorm import Layernorm

def AddAndNorm(gamma, beta, epsilon, H, W, device):
    layernorm = Layernorm(gamma, beta, epsilon, H, W, device, 1)
    def add_and_norm_(activationa, activationb):
        a_plus_b = gpai.tensor.add(activationa, activationb)
        H = activationa.shape()[2]
        lnorm_a_plus_b = layernorm(a_plus_b, overrideH=H)
        return lnorm_a_plus_b

    return add_and_norm_
