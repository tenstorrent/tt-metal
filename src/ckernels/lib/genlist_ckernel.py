#!/usr/bin/python3

#
# Generates C++ include code from list of ckernels. This, along with defines, alows us to build various
# combinations of tensix kernels.
#

import yaml
import cgi
import sys
from pathlib import Path

kernel_list = sys.argv[1]
hlkc_kernel_list = sys.argv[2]
output_file_func = sys.argv[3]
#output_hlkc_file_func = sys.argv[4]
output_file_enum = sys.argv[4]
output_file_string = sys.argv[5]
output_file_make = sys.argv[6]
output_file_gtkw_ckernel = sys.argv[7]
output_file_bazel = sys.argv[8]
arch = sys.argv[9].upper()

def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

def namespace_name(kernel, data):
    # data = data.replace("=", "_")
    # data = data.replace(" ", "_")
    # #These shouldn't do anything, but just in case..
    # kernel = kernel.replace("=", "_")
    # kernel = kernel.replace(" ", "_")
    # return "c" + kernel + "_" +data
    return "c" + kernel

def kernel_file_name(kernel, data):
    return "c" + kernel

def kernel_true_file_name(kernel, data):
    data_items = data.split()
    return "c" + data_items[0]

def kernel_main_func(kernel, data):
    kernel_items = kernel.split("_")
    return kernel_items[1] + "_main"


def gen_output_list(doc, hlkc: bool):

    inc = """

//
// Auto-generated file, do not modify!
//

#pragma once

#include "fw_debug.h"
#include "ckernel_enum.h"
#include "ckernel.h"
#include "ckernel_gpr_map.h"

using namespace ckernel;

    """
    inc += "\n"

    for kernel, data_full in sorted(doc.items()):
        inc += f"#ifdef UCK_C{kernel.upper()}\n"
        if hlkc:
            inc += f"#include \"{kernel_file_name(kernel, data_full)}.cpp\"\n"
            inc += f"namespace {namespace_name(kernel, data_full)} {{ void {kernel_main_func(kernel, data_full)}(const struct hlk_args_t* hlk_args=nullptr, const void* llk_args=nullptr, const int loop_count=0); }}\n"
        else:
            # inc += f"#include \"{kernel_true_file_name(kernel, data_full)}.h\"\n"
            inc += f"namespace {namespace_name(kernel, data_full)} {{ uint kernel_main(uint* params=nullptr); }}\n"
        inc += "#endif\n\n"

    inc += """

uint run_kernel() {
    """
    inc += "\n"

    for kernel, data_full in sorted(doc.items()):
        inc += f"#ifdef UCK_C{kernel.upper()}\n"
        inc += f"    FWLOG1(\"run_kernel = %s\", {kernel.upper()});\n"
        inc += f"    regfile[p_gpr::DBG_CKID] = {kernel.upper()};\n"
        if hlkc:
            inc += f"    {namespace_name(kernel, data_full)}::{kernel_main_func(kernel, data_full)}(&hlk_args, &llk_args, arg_loop_count);\n"
        else:
            inc += f"    return {namespace_name(kernel, data_full)}::kernel_main(nullptr);\n"
        inc +=  "#endif\n\n"

    inc += ""
    inc += "return 0;"
    inc += """

}

    """
    return inc

doc = load_yaml(kernel_list)
file_string = gen_output_list(doc, False)
text_file = open(output_file_func, "w")
text_file.write(file_string)
text_file.close()

#doc_hlkc = load_yaml(hlkc_kernel_list)
#file_string = gen_output_list(doc_hlkc, True)
#text_file = open(output_hlkc_file_func, "w")
#text_file.write(file_string)
#text_file.close()

gtkw_ckernels = """
#
# GTKWave encoding for ckernels from GPR63 value
#
"""

inc = """
//
// Auto-generated file, do not modify!
//

#pragma once
namespace ckernel {


enum kernel_function_id_e {
    FIRMWARE_MSG = 0x8FF,   // Special kernel ID that sends a message to Trisc
    DEBUG_MSG0   = 0x1,     // Debug message to Trisc
    HLKC_UNPACK  = 0x4,     // HLKC generated kernels
    HLKC_MATH    = 0x5,     // HLKC generated kernels
    HLKC_PACK    = 0x6,     // HLKC generated kernels
"""
initial_index =  10 # reserve the first 10 for special values
index = initial_index
for kernel, data_full in sorted(doc.items()):
    first = (index == 0)
    last  = (index == len(doc.items()) - 1 + initial_index)
    inc += f"    {kernel.upper()} = {index}{',' if not last else ''}\n"
    empty = ""
    gtkw_ckernels += f"{hex(index)[2:].zfill(8)} {kernel}\n"
    index += 1

inc += """
};

}
"""

text_file = open(output_file_enum, "w")
text_file.write(inc)
text_file.close()

text_file = open(output_file_gtkw_ckernel, "w")
text_file.write(gtkw_ckernels)
text_file.close()

inc = ""
inc += """
//
// Auto-generated file, do not modify!
//

// This file defines string-to-enum conversion (and back) for all C kernels
// This should be used by test generation only, don't include in C kernels

#pragma once

#include <string>
#include <stdexcept>
#include "ckernel_enum.h"

namespace ckernel {

static std::string function_id_to_string(kernel_function_id_e id)
{
    switch(id) {
"""
for kernel, data_full in sorted(doc.items()):
    inc += f"      case {kernel.upper()} : return \"{kernel}\";\n"

inc += """
        case  FIRMWARE_MSG: return "FIRMWARE_MSG";
        case  DEBUG_MSG0: return "DEBUG_MSG0";
        case  HLKC_UNPACK: return "HLKC_UNPACK";
        case  HLKC_MATH: return "HLKC_MATH";
        case  HLKC_PACK: return "HLKC_PACK";
    }
    return "N/A";
}

static kernel_function_id_e string_to_function_id(const std::string &name)
{
"""
    
for kernel, data_full in sorted(doc.items()):
    inc += f'    if (name == "{kernel}") return {kernel.upper()};\n'

inc += """
        if (name == "FIRMWARE_MSG") return FIRMWARE_MSG;
        if (name == "DEBUG_MSG0") return DEBUG_MSG0;
    throw std::runtime_error("Unknown ckernel: " + name);
}

}

"""

text_file = open(output_file_string, "w")
text_file.write(inc)
text_file.close()

make = """
#
# Auto-generated file, do not modify!
#

.PHONY: all_kernels
"""

bazel = """
#
# Auto-generated file, do not modify!
#
# This is pretty bad... FIXME eventually so we use a repo rule or something 


def kernel_list_translation():
    return {"""




make += "all_kernels: "
for kernel, data_full in sorted(doc.items()):
    make += f"$(OUTPUT_DIR)/c{kernel}.o "

make += "\n\n"

for kernel, data_full in sorted(doc.items()):
    data = data_full.split()
    kernel_cc = data.pop(0)

    defines = ""
    for d in data:
        defines += f" -D{d}"

    #Always set the NAMESPACE
    defines += " -DNAMESPACE=" + namespace_name(kernel, data_full)
    defines += " -DARCH_" + arch

    make += f"""
$(OUTPUT_DIR)/c{kernel}.o: c{kernel_cc}.cc | $(OUTPUT_DIR)
	@echo "CXX $< {defines}"
	$(CXX) $(CXXFLAGS) {defines} -c -o $@ $<

$(OUTPUT_DIR)/c{kernel}.d: c{kernel_cc}.cc | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) {defines} -c -o $@ $<

$(OUTPUT_DIR)/c{kernel}: $(OUTPUT_DIR)/c{kernel}.o

"""
    bazel += f"""
        "c{kernel}.o" : [ "c{kernel_cc}.cc",  {defines.split()} ] ,    """
bazel += """
    }"""
text_file = open(output_file_make, "w")
text_file.write(make)
text_file.close()

bazel_file = open(output_file_bazel, "w")
bazel_file.write(bazel)
bazel_file.close()

################################ NEW #######################################################

# inc = """
# //
# // Auto-generated file, do not modify!
# //

# #pragma once
# using namespace ckernel;

# #include "fw_debug.h"
# #include "ckernel_enum.h"

# """

# for kernel, data_full in sorted(doc.items()):
#     # inc += f"#ifdef UCK_C{kernel.upper()}\n"
#     inc += f"namespace {namespace_name(kernel, data_full)} {{ uint kernel_main(uint *params = nullptr); }}\n"
#     # inc += "#endif\n\n"

# inc += """

# uint run_kernel(kernel_function_id_e function_id, uint *params) {

#     FWLOG1("run_kernel = %d", function_id);
#     switch (function_id) {
# """

# for kernel, data_full in sorted(doc.items()):
#     # inc += f"        #ifdef UCK_C{kernel.upper()}\n"
#     inc += f"            case {kernel.upper()}: return {namespace_name(kernel, data_full)}::kernel_main(params);\n"
#     # inc +=  "        #endif\n\n"

# inc += """
#         default:
#             FWASSERT("Unknown kernel function", 0);
#     }        
#     return 0;
# }

# """

# text_file = open("/localhome/shuang/tensix/src/ckernels/gen/out/ckernel_list_new.h", "w")
# text_file.write(inc)
# text_file.close()


# Auto generate blank .cc files

blank_kernel_src_contents = """
// Auto-generated file

#include "ckernel.h"

namespace NAMESPACE
{

uint kernel_main(uint *params = nullptr)
{
    return 0;
}

} // namespace NAMESPACE
"""

blank_srcs_output_dir = Path(output_file_func).parent / "blank"
blank_srcs_output_dir.mkdir(exist_ok=True)

for kernel, data_full in sorted(doc.items()):
    data = data_full.split()
    data = data.pop(0)
    blank_kernel_src_name = f"c{data}.cc"
    text_file = open(f"{blank_srcs_output_dir}/{blank_kernel_src_name}", "w")
    text_file.write(blank_kernel_src_contents)
    text_file.close()
