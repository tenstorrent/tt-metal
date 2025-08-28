#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to generate RpcServer class from Cap'n Proto schema.
Creates a callback-based RPC server implementation.
"""

import re
import sys
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MethodInfo:
    name: str
    ordinal: int
    params: List[Tuple[str, str]]  # (name, type)
    return_type: str
    return_name: str





def parse_capnp_interface(capnp_file: str) -> List[MethodInfo]:
    """Parse Cap'n Proto schema file and extract interface methods."""
    
    with open(capnp_file, 'r') as f:
        content = f.read()
    
    # Remove comments to avoid parsing issues
    content = re.sub(r'#[^\n]*', '', content)
    
    # Find the interface block
    interface_match = re.search(r'interface\s+(\w+)\s*\{([^}]+)\}', content, re.DOTALL)
    if not interface_match:
        raise ValueError("No interface found in Cap'n Proto file")
    
    interface_name = interface_match.group(1)
    interface_body = interface_match.group(2)
    
    methods = []
    
    # Parse method definitions - handle multiline methods
    # Match pattern: methodName @ordinal (params) -> (returns);
    # First normalize whitespace to handle multiline definitions
    interface_body = re.sub(r'\s+', ' ', interface_body)
    
    # Updated pattern to handle List(...) return types properly
    method_pattern = r'(\w+)\s+@(\d+)\s*\(([^)]*)\)\s*->\s*\(([^;]+?)\)\s*;'
    
    # Find all matches
    all_matches = list(re.finditer(method_pattern, interface_body))
    
    for match in all_matches:
        method_name = match.group(1)
        ordinal = int(match.group(2))
        params_str = match.group(3).strip()
        return_str = match.group(4).strip()
        
        # Parse parameters
        params = []
        if params_str:
            # Split by comma, but be careful about nested types
            param_parts = []
            paren_count = 0
            current_param = ""
            
            for char in params_str:
                if char == '(' or char == '<':
                    paren_count += 1
                elif char == ')' or char == '>':
                    paren_count -= 1
                elif char == ',' and paren_count == 0:
                    param_parts.append(current_param.strip())
                    current_param = ""
                    continue
                current_param += char
            
            if current_param.strip():
                param_parts.append(current_param.strip())
            
            for param in param_parts:
                if param:
                    parts = param.split(':')
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        param_type = parts[1].strip()
                        params.append((param_name, param_type))
        
        # Parse return type
        return_parts = return_str.split(':')
        if len(return_parts) == 2:
            return_name = return_parts[0].strip()
            return_type = return_parts[1].strip()
        else:
            return_name = "result"
            return_type = return_str
            
        methods.append(MethodInfo(method_name, ordinal, params, return_type, return_name))
    
    return methods


def generate_header(methods: List[MethodInfo]) -> str:
    """Generate the header file content."""
    
    header = '''// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Auto-generated file - DO NOT EDIT
// Generated from inspector.capnp by generate_inspector_rpc_server.py

#pragma once

#include <functional>
#include <capnp/ez-rpc.h>
#include "impl/debug/inspector/rpc.capnp.h"

namespace tt::tt_metal::inspector {

class RpcServer final : public rpc::Inspector::Server {
public:
    RpcServer() = default;
    ~RpcServer() = default;

    // RPC interface implementation - delegates to callbacks
'''

    # Generate override methods  
    for method in methods:
        # Use full CallContext type since Server::Context might not exist
        method_name_cap = method.name[0].upper() + method.name[1:]
        context_name = f"::capnp::CallContext<rpc::Inspector::{method_name_cap}Params, rpc::Inspector::{method_name_cap}Results>"
        header += f'''    ::kj::Promise<void> {method.name}({context_name} context) override;
'''

    header += '''
    // Callback setters
'''

    # Generate callback type aliases and setters
    for method in methods:
        # Create proper Cap'n Proto types
        method_name_cap = method.name[0].upper() + method.name[1:]
        results_builder = f"rpc::Inspector::{method_name_cap}Results::Builder"
        callback_type_name = f"{method_name_cap}Callback"
        setter_name = f"set{method_name_cap}Callback"
        
        # Build callback signature based on whether method has parameters
        if method.params:
            params_reader = f"rpc::Inspector::{method_name_cap}Params::Reader"
            callback_signature = f"void({params_reader} params, {results_builder} results)"
        else:
            callback_signature = f"void({results_builder} results)"
        
        header += f'''    using {callback_type_name} = std::function<{callback_signature}>;
    void {setter_name}(const {callback_type_name}& callback) {{
        {method.name}_callback = callback;
    }}
'''

    header += '''
private:
'''

    # Generate callback member variables
    for method in methods:
        callback_type_name = method.name[0].upper() + method.name[1:] + "Callback"
        header += f'''    {callback_type_name} {method.name}_callback;
'''

    header += '''};

} // namespace tt::tt_metal::inspector
'''

    return header


def generate_source(methods: List[MethodInfo]) -> str:
    """Generate the source file content."""
    
    source = '''// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Auto-generated file - DO NOT EDIT
// Generated from impl/debug/inspector/rpc.capnp by generate_inspector_rpc_server.py

#include "impl/debug/inspector/rpc_server_generated.hpp"
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::inspector {

'''

    # Generate implementation methods
    for method in methods:
        # Use full CallContext type to match override signature
        method_name_cap = method.name[0].upper() + method.name[1:]
        context_name = f"::capnp::CallContext<rpc::Inspector::{method_name_cap}Params, rpc::Inspector::{method_name_cap}Results>"
        
        # Build callback call based on whether method has parameters
        if method.params:
            callback_call = f"{method.name}_callback(context.getParams(), context.getResults());"
        else:
            callback_call = f"{method.name}_callback(context.getResults());"
        
        source += f'''::kj::Promise<void> RpcServer::{method.name}({context_name} context) {{
    try {{
        if (!{method.name}_callback) {{
            log_error(tt::LogInspector, "No callback set for {method.name}");
            return ::kj::READY_NOW;
        }}
        {callback_call}
        return ::kj::READY_NOW;
    }} catch (const std::exception& e) {{
        log_warning(tt::LogInspector, "Failed to execute {method.name}: {{}}", e.what());
        return kj::Promise<void>(KJ_EXCEPTION(FAILED, e.what()));
    }}
}}

'''

    source += '''} // namespace tt::tt_metal::inspector
'''

    return source


def main():
    if len(sys.argv) != 4:
        print("Usage: generate_inspector_rpc_server.py <capnp_file> <output_header> <output_source>")
        sys.exit(1)
    
    capnp_file = sys.argv[1]
    output_header = sys.argv[2]
    output_source = sys.argv[3]
    
    try:
        methods = parse_capnp_interface(capnp_file)
        
        # Generate header
        header_content = generate_header(methods)
        with open(output_header, 'w') as f:
            f.write(header_content)
        
        # Generate source
        source_content = generate_source(methods)
        with open(output_source, 'w') as f:
            f.write(source_content)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
