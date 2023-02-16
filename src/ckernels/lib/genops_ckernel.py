#!/usr/bin/python3

#
# Generates C++ code from assembly.yaml
#

import yaml
import cgi
import sys

def has_arguments(data):
    if not "arguments" in data or data["arguments"] in (None, 0):
        return False

    return True

def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

doc = load_yaml(sys.argv[1])

# 
# - Generate macros for Trisc to issue instructions
##

inc = """
//
// Auto-generated file, do not modify!
//

#pragma once

#define TT_OP(opcode, params) ( (opcode << 24) + params )
#define INSTRUCTION_WORD(x) __asm__ __volatile__(".word (%0)" : : "i" ((x))) // Drop 32 bits into the instruction stream.
#define TRISC_OP_SWIZZLE(x) ( (((x) >> 30) & 0x3) | (((x) & 0x3FFFFFFF) << 2) ) // Put top 2 bits, which are currently never 'b11 to bottom, indicating to Risc that they are not risc instructions

"""

for instr, data in sorted(doc.items()):

    # SETC is special
    if instr == "SETC":
        inc += f"""
#define TT_SETC(set_value, set_reg) \
    ckernel::instrn_buffer[0] = TT_OP(0x{format(data["op_binary"], '02x')}, set_reg); \
    ckernel::instrn_buffer[0] = TT_OP(0, set_value); 
#define TTI_SETC(set_value, set_reg) \
    INSTRUCTION_WORD(TRISC_OP_SWIZZLE(TT_OP(0x{format(data["op_binary"], '02x')}, set_reg))); \
    INSTRUCTION_WORD(TRISC_OP_SWIZZLE(TT_OP(0, set_value)));

"""
        continue

    inc += "#define TT_OP_" + instr
    params = ""
    checks = ""
    if has_arguments(data):
        params += "("
        for index, arg in enumerate(reversed(data["arguments"])):
            name = arg["name"] if "name" in arg else "arg" + str(index)
            name = name.replace(" ", "_")
            if index > 0:
                params += ", "
            params += name
        params += ") "
    inc += params
    inc += "\\\n"

    inc += "  TT_OP(0x" + format(data["op_binary"], '02x') 

    if has_arguments(data):
        offset = 24
        inc += ", ("
        for index, arg in enumerate(reversed(data["arguments"])):
            name = arg["name"] if "name" in arg else "arg" + str(index)
            name = name.replace(" ", "_")
            start = int(arg["start_bit"])
            if index > 0:
                inc += " + "
                checks += " && "
            inc += "((" + name + ") << " + str(start) + ")"
            checks += "ckernel::is_valid(" + arg["name"] + ", " + str(offset - start) + ")"
            offset = start
        inc += ")"
    else:
        inc += ", 0"
    
    inc += ")\n"

    # Follow the principle of defining errors out of existence:
    # only support TT macro for instructions that have arguments
    if has_arguments(data):
        inc += "#define TT_" + instr + "_VALID" + params + "\\\n"
        inc += "  (" + checks + ")\n"
        inc += "#define TT_" + instr + params + "\\\n"
        inc += "  ckernel::instrn_buffer[0] = TT_OP_" + instr + params + "\n"

    inc += "#define TTI_" + instr + params + "\\\n"
    inc += "  INSTRUCTION_WORD(TRISC_OP_SWIZZLE(TT_OP_" + instr + params + "))\n"
    inc += "\n"

text_file = open(sys.argv[2], "w")
text_file.write(inc)
text_file.close()

