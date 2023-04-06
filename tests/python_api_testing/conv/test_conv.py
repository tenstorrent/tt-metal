# Work in Progress
import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np
import torch
from python_api_testing.conv.pytorch_conv_tb import ConvTestParameters, generate_pytorch_golden_tb, generate_conv_tb

if __name__ == "__main__":
    print("Sweep over convolution sizes and parameters in conv_tb.yaml.")
    print("Generate testbench. Run pytorch convolution for golden.")
    test_bench = generate_conv_tb()
    pytorch_conv_golden_tb = generate_pytorch_golden_tb(test_bench)
    print("Total number of tests - " + str(len(test_bench)))
