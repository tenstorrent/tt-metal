#!/usr/bin/python3

#
# Generates CMakeLists.txt with libraries for each ckernel
#

import yaml
import cgi
import sys
import os

kernel_list = sys.argv[1]
output_file_src = sys.argv[2]
# output_file_test = sys.argv[3]

def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

def namespace_name(kernel, data):
    data = data.replace("=", "_")
    data = data.replace(" ", "_")
    return "c" + data
    #return "c" + kernel


doc = load_yaml(kernel_list)

header = """
#
# Auto-generated file, do not modify!
#

cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-D TENSIX_FIRMWARE)
add_definitions(-m32)


"""

src_cmake = header
# test_cmake = header

for kernel, data_full in sorted(doc.items()):


    kernel_build = "ckernel_" + kernel;

    data = data_full.split()
    kernel_cc = data.pop(0)

    c_file = "c" + kernel_cc + ".cc"

    defines = ""
    for d in data:
        defines += f" {d}"

    #Always set the NAMESPACE
    defines += " NAMESPACE=" + namespace_name(kernel, data_full)

    inc = f"add_library({kernel_build} {c_file})\n"
    if defines != "":
        inc += f"target_compile_definitions({kernel_build} PRIVATE {defines})\n"

    inc += f"target_include_directories({kernel_build} PRIVATE ../$ENV""{ARCH_NAME}""/common/inc ../gen/out $ENV""{ROOT}""/src/firmware/riscv/common)\n"
    inc += "\n"

    src_cmake += inc

text_file = open(output_file_src, "w")
text_file.write(src_cmake)
text_file.close()
# text_file = open(output_file_test, "w")
# text_file.write(test_cmake)
# text_file.close()
