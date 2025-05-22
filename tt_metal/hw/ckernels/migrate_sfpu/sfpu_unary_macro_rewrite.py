# /// script
# dependencies = [
#   "tree-sitter",
#   "tree-sitter-cpp",
# ]
# ///

from pathlib import Path
from typing import List
from typing import Optional
import argparse
import re
import shutil
import subprocess
import tree_sitter
import tree_sitter_cpp

skipped_files = [
    "cumsum.h",
    "erf_erfc.h",
    "exp.h",
    "gelu.h",
    "init.h",
    "int_sum.h",
    "mask.h",
    "params.h",
    "sigmoid_appx.h",
    "typecast.h",
]


def main():
    parser = argparse.ArgumentParser(
        prog="sfpu_unary_macro_rewrite", description="Rewrite sfpu eltwise unary functions into macros"
    )
    parser.add_argument("input_dir", help="Path of folder containing the llk header files to refactor")
    parser.add_argument("output_dir", help="Path to folder to store outputs")
    parser.add_argument("--macros", help="Path to file defining helper macros")
    args = parser.parse_args()

    assert args.input_dir != args.output_dir
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.macros is None:
        macro_file = Path(__file__).parent / "llk_math_eltwise_unary_sfpu_macros.h"
    else:
        macro_file = Path(args.macro_file)

    output_dir.mkdir(exist_ok=True)
    
    shutil.copyfile(macro_file, output_dir / "llk_math_eltwise_unary_sfpu_macros.h")

    handlers = [handle_include, handle_init_fn, handle_calculate_fn]

    for f in sorted(input_dir.iterdir()):
        # Copy unused files over unchanged
        if not f.name.startswith("llk_math_eltwise_unary_sfpu"):
            shutil.copy(f, output_dir / f.name)
            continue
        if f.name.removeprefix("llk_math_eltwise_unary_sfpu_") in skipped_files:
            shutil.copy(f, output_dir / f.name)
            continue
            
        out = open(output_dir / f.name, "wb") 
        original = open(f, "rb").read()
        # Parse the syntax tree, swapping out the parts of the file where we have re-writes available 
        tree = tree_sitter.Parser(CPP_LANGUAGE).parse(open(f, "rb").read())
        nodes = [next(iter(d.values()))[0] for _, d in QUERY.matches(tree.root_node)]
        last_byte = 0
        for n in sorted(nodes, key=lambda x: x.start_byte):
            out.write(original[last_byte: n.start_byte])
            last_byte = n.end_byte
            found_handler = False
            for h in handlers:
                if h(n) is not None:
                    assert valid_rewrite(n, h(n), macro_file)
                    if h(n).startswith("#include"):
                        out.write(b'#include "llk_math_eltwise_unary_sfpu_macros.h"\n')
                    out.write(h(n).encode())
                    found_handler = True
                    break
            assert found_handler
        out.write(original[last_byte:])

PREAMBLE = """\
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC and TT-metal contributors
// 
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

"""

CPP_LANGUAGE = tree_sitter.Language(tree_sitter_cpp.language())

# Simple query to find function definitions and relevant
# ckernel header includes
QUERY = CPP_LANGUAGE.query("""
(template_declaration (function_definition)) @function
    (declaration_list (function_definition) @function)
    ((preproc_include) @include_path
        (#match? @include_path "ckernel_sfpu_.*.h"))
""")

# Query to help pull out argument lists of interest from compute functions
# Groups:
#   - template_params: Outer-most template parameters
#   - fn_name: Outer-most function name
#   - fn_params: Outer-most function parameters
#   - ckernel_name: Name of the ckernel::sfpu::calculate_* function
#                   (dropping the ckernel::sfpu part)
#   - template_args: Template arguments passed to ckernel
#   - fn_args: Arguments passed to ckernel
CALCULATE_FN_QUERY = CPP_LANGUAGE.query("""
(template_declaration 
	parameters: (_) @template_params
        (function_definition
        	declarator: (function_declarator
            	declarator: (_) @fn_name
                parameters: (_) @fn_params)
            body: (compound_statement (expression_statement (call_expression
                arguments: (argument_list 
                	. (qualified_identifier name: (qualified_identifier name:
                    	(template_function
                        	name: (_) @ckernel_name
                            arguments: (_) @template_args)))
                    )@fn_args )))))
""")

# Query to help pull out arguments of interest from *_init functions
# Groups:
#   - template_params: Top-level template parameters
#   - fn_name: Top-level function name
#   - fn_params: Top-level parameters
#   - template_args: Template arguments passed to llk_math_eltwise_unary_sfpu_init
#   - fn_args: arguments passed to llk_math_eltwise_unary_init
INIT_FN_QUERY = CPP_LANGUAGE.query("""
(template_declaration 
	parameters: (_) @template_params
            (function_definition
            	declarator: (function_declarator
                	declarator: (_) @fn_name
                    parameters: (_) @fn_params)
                body: (compound_statement (expression_statement (call_expression
                    function: (template_function
                    	arguments: (_) @template_args)
                    arguments: (_) @fn_args)))))
""")


def parse_functions(prog: str) -> List[tree_sitter.Node]:
    parser = tree_sitter.Parser(CPP_LANGUAGE)
    tree = parser.parse(prog.encode())
    return sorted(QUERY.captures(tree.root_node)["function"], key=lambda x: x.start_byte)


def fn_name(fn: tree_sitter.Node) -> str:
    if fn.type == "template_declaration":
        fn = fn.child(2)
    assert fn.type == "function_definition"
    return fn.child_by_field_name("declarator").child_by_field_name("declarator").text.decode()


def handle_include(node: tree_sitter.Node) -> Optional[str]:
    if node.type == "preproc_include":
        return node.text.decode()


def handle_init_fn(fn: tree_sitter.Node) -> Optional[str]:
    capts = INIT_FN_QUERY.captures(fn)

    # Check function name and decode op name
    fn_name = capts["fn_name"][0].text.decode()
    fn_name_match = re.match("llk_math_eltwise_unary_sfpu_([a-z0-9_]+)_init", fn_name)
    if fn_name_match is None:
        return
    op = fn_name_match.group(1)

    # Check template params
    template_params = capts["template_params"][0].named_children
    assert len(template_params) == 1
    assert template_params[0].text == b"bool APPROXIMATE"

    # Check template args
    template_args = capts["template_args"][0].named_children
    assert len(template_args) == 2
    assert template_args[-1].text == b"APPROXIMATE"

    # Check function args and params
    fn_args = capts["fn_args"][0].named_children
    fn_params = capts["fn_params"][0].named_children

    # Check op type name
    assert template_args[0].text.decode().startswith("SfpuType::")
    op_type = template_args[0].text.decode().removeprefix("SfpuType::")

    # Decide which macro variant to use
    if len(fn_args) == 0 and op == op_type:
        return f"SFPU_INIT({op})"
    if len(fn_args) == 0:  # (op != op_type)
        return f"SFPU_INIT_CUSTOM_NAME({op}, {op_type})"

    # len(fn_args) > 0
    fn_name = fn_args[0].text.decode()
    assert fn_name.startswith("sfpu::")
    assert fn_name.endswith("<APPROXIMATE>")
    fn_name = fn_name.removeprefix("sfpu::").removesuffix("<APPROXIMATE>")

    if op == op_type and fn_name == f"{op}_init":
        return f"SFPU_INIT_WITH_FN({op}{handle_params(fn_params, fn_args[1:])})"
    return f"SFPU_INIT_CUSTOM_NAME_WITH_FN({op}, {op_type}, {fn_name}{handle_params(fn_params, fn_args[1:])})"


def handle_params(params: List[tree_sitter.Node], args: List[tree_sitter.Node], leading_comma=True):
    i_arg = 0
    ret = ""
    for i_param in range(len(params)):
        p = params[i_param]
        type = p.child_by_field_name("type").text.decode()
        name = p.child_by_field_name("declarator").text.decode()
        if p.type == "optional_parameter_declaration":
            val = p.child_by_field_name("default_value").text.decode()
        while i_arg < len(args):
            a = args[i_arg]
            i_arg += 1
            if a.text.decode() == name:
                break
            ret += f", ARG({a.text.decode()})"
        assert i_arg <= len(args)
        if p.type == "parameter_declaration":
            ret += f", PARAM({type}, {name})"
        elif p.type == "optional_parameter_declaration":
            ret += f", DEFAULT_PARAM({type}, {name}, {val})"
        else:
            assert False
    while i_arg < len(args):
        a = args[i_arg]
        ret += f", ARG({a.text.decode()})"
        i_arg += 1
    if not leading_comma:
        ret = ret[2:]
    return ret


def handle_calculate_fn(fn: tree_sitter.Node) -> Optional[str]:
    capts = CALCULATE_FN_QUERY.captures(fn)

    # Check function name and decode op name
    fn_name = capts["fn_name"][0].text.decode()
    fn_name_match = re.match("llk_math_eltwise_unary_sfpu_([a-z0-9_]+)", fn_name)
    if fn_name_match is None:
        return
    op = fn_name_match.group(1)

    # Check template parameters
    template_params = capts["template_params"][0].named_children
    assert template_params[0].text == b"bool APPROXIMATE"
    template_params = template_params[1:]

    # Check function parameters
    fn_params = capts["fn_params"][0].named_children
    assert fn_params[0].text in [b"uint dst_index", b"uint32_t dst_index"]
    has_vector_mode = fn_params[-1].text.removeprefix(b"int vector_mode = ") in [
        b"(int)VectorMode::RC",
        b"VectorMode::RC",
        b"(int)VectorMode::RC_custom",
    ]
    if not has_vector_mode:
        vector_mode = "ALWAYS_RC"
    elif "RC_custom" in fn_params[-1].text.decode():
        vector_mode = "RC_CUSTOM"
    else:
        vector_mode = "RC"

    if has_vector_mode:
        fn_params = fn_params[1:-1]
    else:
        fn_params = fn_params[1:]

    # Check ckernel name
    ckernel_name = capts["ckernel_name"][0].text.decode()

    # Check template args
    template_args = capts["template_args"][0].named_children
    if template_args[0].text != b"APPROXIMATE":
        assert template_args[-1].text == b"APPROXIMATE"
        approx_last = "_APPROX_LAST"
        template_args = template_args[:-1]
    else:
        assert template_args[0].text == b"APPROXIMATE"
        approx_last = ""
        template_args = template_args[1:]

    # Check function args
    fn_args = capts["fn_args"][0].named_children[1:]
    assert len(fn_args) >= 2
    assert fn_args[0].text == b"dst_index"
    if has_vector_mode:
        assert fn_args[1].text == b"vector_mode"
        irregular_vector_mode = False
        fn_args = fn_args[2:]
    elif  b"VectorMode::RC" not in fn_args[1].text:
        irregular_vector_mode = True
        fn_args = fn_args[1:]
    else:
        irregular_vector_mode = False
        fn_args = fn_args[2:]

    if len(template_args) == 0 and ckernel_name == f"calculate_{op}" and vector_mode == "RC":
        return f"SFPU_CALCULATE({op}{handle_params(fn_params, fn_args)})"
    elif irregular_vector_mode:
        return (
            f"SFPU_CALCULATE_MANUAL{approx_last}({op}, {ckernel_name}, PARAM_LIST(),"
            + f"PARAM_LIST({handle_params(template_params, template_args, False)}), "
            + f"PARAM_LIST({handle_params(fn_params, fn_args, False)}))"
        )
    return (
        f"SFPU_CALCULATE_{vector_mode}{approx_last}({op}, {ckernel_name}, "
        + f"PARAM_LIST({handle_params(template_params, template_args, False)}), "
        + f"PARAM_LIST({handle_params(fn_params, fn_args, False)}))"
    )


def valid_rewrite(original: tree_sitter.Node, transformed: str, macro_file: Path) -> bool:
    if transformed.startswith("#include"):
        return original.text.decode().strip() == transformed.strip()

    transformed_raw = transformed

    original = original.text.decode()
    # Handle slight inconsistencies in original sources
    original = (
        original.replace("int vector_mode = VectorMode::RC", "int vector_mode = (int)VectorMode::RC")
        .replace("uint32_t dst_index", "uint dst_index")
        .replace(", VectorMode::RC", ", (int)VectorMode::RC")
    )

    original = re.sub(r"\s*//[^\n]+\n", "", original)

    transformed = f"""
    #include "{str(macro_file)}"
    {transformed}
    """
    transformed = clang_format(preprocess_macros(transformed))
    original = clang_format(original)
    if transformed != original:
        print(
            f"/* ERROR transformed != original\n\nORIGINAL\n\n{original}\n\nTRANSFORMED\n\n{transformed_raw}\n\nTRANFORMED (Macro expanded)\n\n{transformed}\n*/"
        )
    return transformed == original


def clang_format(prog: str) -> str:
    return subprocess.run(["clang-format"], input=prog.encode(), stdout=subprocess.PIPE).stdout.decode().strip()


def preprocess_macros(prog: str) -> str:
    prog = subprocess.run(
        ["g++", "-E", "-x", "c++", "-"],
        input=prog.encode(),
        stdout=subprocess.PIPE,
    ).stdout.decode()
    prog = re.sub(r"#[^\n+]*\n", "", prog)
    prog = re.sub("\n+", "\n", prog)
    return prog


if __name__ == "__main__":
    main()
