#!/usr/bin/python3

#
# Generates a full_kernel_list.yaml that includes all the kernels
# from kernel_list.yamls, plus empty versions.
#

import yaml
import cgi
import sys


kernel_list = sys.argv[1]
output_full_kernel_list = sys.argv[2]


def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

doc = load_yaml(kernel_list)

inc = """
#
# Auto-generated file, do not modify!
#
"""
for kernel, data_full in sorted(doc.items()):
    inc += f"{kernel}: {data_full}\n"
    data = data_full.split()
    kernel_cc = data.pop(0) + "_blank"
    data = [kernel_cc] + data
    data_full = " ".join(data)
    inc += f"{kernel}_blank: {data_full}\n"

text_file = open(output_full_kernel_list, "w")
text_file.write(inc)
text_file.close()
