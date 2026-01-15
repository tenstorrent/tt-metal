import sys
import torch
import pathlib

sys.path.insert(0, str(pathlib.Path.cwd()))

from common import BITS_INFO, float_info_mapping
from common import create_float, parse_float, parse_float_value


if __name__ == "__main__":
    # 两个数相加，会首先按照大数的exponent进行对齐，按照exponent
    left_value = create_float(0, 127, 64, torch.bfloat16).cuda()
    for exponent in range(127, 118, -1):
        right_value = create_float(0, exponent, 64, torch.bfloat16).cuda()
        output_value = left_value + right_value
        left_results = parse_float_value(left_value)
        right_results = parse_float_value(right_value)
        output_results = parse_float_value(output_value)
        print(f"{left_value.item()} + {right_value.item()} = {output_value.item()}")
        print(f"{output_results[0]}, {bin(output_results[1])}")
        print("")
