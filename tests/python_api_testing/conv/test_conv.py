# Work in Progress
import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np
import torch
from python_api_testing.conv.pytorch_conv_tb import ConvTestParameters, generate_pytorch_golden_tb

if __name__ == "__main__":
    print("Sweep over convolution sizes and parameters in conv_tb.yaml.")
    print("Generate testbench. Run pytorch convolution for golden.")
    pytorch_golden_test_bench = generate_pytorch_golden_tb()
    for test in pytorch_golden_test_bench:
        print("Test with following parameters - ")
        test.print("   ")
    print("Total number of tests - " + str(len(pytorch_golden_test_bench)))
