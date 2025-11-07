// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "callstack.hpp"

// Include Python headers if available
// Check for Python 3.10 specifically (the version used in this environment)
#ifdef __has_include
#if __has_include(<python3.10/Python.h>)
#include <python3.10/Python.h>
#include <python3.10/frameobject.h>
#define HAS_PYTHON_HEADERS 1
#elif __has_include(<Python.h>)
#include <Python.h>
#include <frameobject.h>
#define HAS_PYTHON_HEADERS 1
#endif
#endif

// Include dynamic loading support for runtime Python detection
#include <dlfcn.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <vector>

namespace tt::tt_metal::inspector {

std::string get_python_callstack() {
    // Try to get Python callstack if we're called from Python
    // Use dlsym to dynamically load Python functions at runtime
    // This avoids linking against Python libraries for pure C++ builds

    // Function pointers for Python API
    static auto* py_is_initialized = (int (*)())dlsym(RTLD_DEFAULT, "Py_IsInitialized");
    if (!py_is_initialized || !py_is_initialized()) {
        return "";  // Python not available
    }

#ifdef HAS_PYTHON_HEADERS
    // Define types for Python API functions we need
    using PyGILState_Ensure_t = PyGILState_STATE (*)();
    using PyGILState_Release_t = void (*)(PyGILState_STATE);
    using PyEval_GetFrame_t = PyFrameObject* (*)();
    using PyFrame_GetCode_t = PyCodeObject* (*)(PyFrameObject*);
    using PyFrame_GetBack_t = PyFrameObject* (*)(PyFrameObject*);
    using PyFrame_GetLineNumber_t = int (*)(PyFrameObject*);
    using PyUnicode_AsUTF8_t = const char* (*)(PyObject*);
    // Note: Py_DECREF is actually a macro in Python, not a function
    // We can't directly load it, so we'll handle refcounting differently

    // Load Python API functions dynamically
    static auto* dyn_PyGILState_Ensure = (PyGILState_Ensure_t)dlsym(RTLD_DEFAULT, "PyGILState_Ensure");
    static auto* dyn_PyGILState_Release = (PyGILState_Release_t)dlsym(RTLD_DEFAULT, "PyGILState_Release");
    static auto* dyn_PyEval_GetFrame = (PyEval_GetFrame_t)dlsym(RTLD_DEFAULT, "PyEval_GetFrame");
    static auto* dyn_PyFrame_GetCode = (PyFrame_GetCode_t)dlsym(RTLD_DEFAULT, "PyFrame_GetCode");
    static auto* dyn_PyFrame_GetBack = (PyFrame_GetBack_t)dlsym(RTLD_DEFAULT, "PyFrame_GetBack");
    static auto* dyn_PyFrame_GetLineNumber = (PyFrame_GetLineNumber_t)dlsym(RTLD_DEFAULT, "PyFrame_GetLineNumber");
    static auto* dyn_PyUnicode_AsUTF8 = (PyUnicode_AsUTF8_t)dlsym(RTLD_DEFAULT, "PyUnicode_AsUTF8");

    // Check if all required functions are available
    if (!dyn_PyGILState_Ensure || !dyn_PyGILState_Release || !dyn_PyEval_GetFrame || !dyn_PyFrame_GetCode ||
        !dyn_PyFrame_GetBack || !dyn_PyFrame_GetLineNumber || !dyn_PyUnicode_AsUTF8) {
        return "";  // Some Python functions not available
    }

    // Define additional function types for safer access to code object attributes
    using PyObject_GetAttrString_t = PyObject* (*)(PyObject*, const char*);
    using Py_DecRef_t = void (*)(PyObject*);

    static auto* dyn_PyObject_GetAttrString = (PyObject_GetAttrString_t)dlsym(RTLD_DEFAULT, "PyObject_GetAttrString");
    static auto* dyn_Py_DecRef = (Py_DecRef_t)dlsym(RTLD_DEFAULT, "Py_DecRef");

    if (!dyn_PyObject_GetAttrString) {
        return "";  // Can't safely access Python objects
    }

    PyGILState_STATE gstate = dyn_PyGILState_Ensure();
    std::stringstream result;  // Store result to return after releasing GIL

    // Keep track of frames we need to clean up
    std::vector<PyFrameObject*> frames_to_clean;

    try {
        PyFrameObject* frame = dyn_PyEval_GetFrame();  // Borrowed reference
        int frame_count = 0;
        const int max_frames = 20;

        if (frame != nullptr) {
            // Traverse up the stack to collect user frames
            while (frame != nullptr && frame_count < max_frames) {
                PyCodeObject* code = dyn_PyFrame_GetCode(frame);
                if (code != nullptr) {
                    // Use PyObject_GetAttrString for thread-safe access to co_filename
                    PyObject* filename_obj = dyn_PyObject_GetAttrString((PyObject*)code, "co_filename");
                    if (filename_obj) {
                        const char* filename = dyn_PyUnicode_AsUTF8(filename_obj);
                        if (filename) {
                            std::string filename_str(filename);
                            // Skip internal frames
                            if (filename_str.find("site-packages") == std::string::npos &&
                                filename_str.find("ttnn/decorators.py") == std::string::npos &&
                                filename_str.find("ttnn/api") == std::string::npos &&
                                filename_str.find("_ttnn.so") == std::string::npos) {
                                // Get line number
                                int lineno = dyn_PyFrame_GetLineNumber(frame);

                                // Make path relative to current working directory
                                std::filesystem::path filepath(filename_str);
                                std::string relative_path;
                                try {
                                    std::filesystem::path cwd = std::filesystem::current_path();
                                    if (filepath.is_absolute()) {
                                        relative_path = std::filesystem::relative(filepath, cwd).string();
                                    } else {
                                        relative_path = filepath.string();
                                    }
                                } catch (...) {
                                    // If relative path computation fails, fall back to just filename
                                    relative_path = filepath.filename().string();
                                }

                                // Add frame with numbering
                                if (frame_count > 0) {
                                    result << " ";
                                }
                                result << "#" << frame_count << " " << relative_path << ":" << lineno;
                                frame_count++;

                                // Clean up the filename object
                                if (dyn_Py_DecRef) {
                                    dyn_Py_DecRef(filename_obj);
                                }

                                // Continue to collect more frames (no break here)
                            } else {
                                // Clean up the filename object for skipped frames too
                                if (dyn_Py_DecRef) {
                                    dyn_Py_DecRef(filename_obj);
                                }
                            }
                        } else {
                            // Always clean up the filename object
                            if (dyn_Py_DecRef && filename_obj) {
                                dyn_Py_DecRef(filename_obj);
                            }
                        }
                    }
                }

                // Move to the next frame
                PyFrameObject* prev_frame = frame;
                PyFrameObject* next_frame = dyn_PyFrame_GetBack(frame);

                // PyFrame_GetBack returns a new reference, add it to cleanup list
                if (next_frame) {
                    frames_to_clean.push_back(next_frame);
                }

                frame = next_frame;

                // Check for infinite loop
                if (frame == prev_frame) {
                    break;
                }

                // Also break if we've traversed too many frames (safety limit)
                if (frames_to_clean.size() > 100) {
                    break;
                }
            }
        }

        // Clean up all frames we got from PyFrame_GetBack (they are new references)
        if (dyn_Py_DecRef) {
            for (PyFrameObject* f : frames_to_clean) {
                if (f) {
                    dyn_Py_DecRef((PyObject*)f);
                }
            }
        }
    } catch (...) {
        // Clean up frames before releasing GIL
        if (dyn_Py_DecRef) {
            for (PyFrameObject* f : frames_to_clean) {
                if (f) {
                    dyn_Py_DecRef((PyObject*)f);
                }
            }
        }
        // Ensure we always release the GIL even if an exception occurs
        dyn_PyGILState_Release(gstate);
        return "";
    }

    dyn_PyGILState_Release(gstate);
    return result.str();
#endif
    return "";  // Return empty string if Python callstack not available
}

std::string get_cpp_callstack() {
    void* buffer[64];
    int nptrs = backtrace(buffer, 64);

    if (nptrs > 1) {
        char** symbols = backtrace_symbols(buffer, nptrs);
        if (symbols != nullptr) {
            std::stringstream callstack;
            int valid_frames = 0;
            const int max_frames = 20;

            // Skip the first 3 frames (this function, get_callstack, and track_operation)
            for (int i = 3; i < nptrs && i < 64 && valid_frames < max_frames; ++i) {
                std::string symbol_str(symbols[i]);

                // Skip internal Inspector frames only (keep decorator frames for now to debug)
                if (symbol_str.find("Inspector::") != std::string::npos ||
                    symbol_str.find("inspector::") != std::string::npos ||
                    symbol_str.find("get_callstack") != std::string::npos ||
                    symbol_str.find("get_cpp_callstack") != std::string::npos ||
                    symbol_str.find("get_python_callstack") != std::string::npos ||
                    symbol_str.find("track_operation") != std::string::npos) {
                    continue;
                }

                // Use the actual address from buffer[i] for addr2line resolution.
                // Format the address as hex for dump_ops.py to resolve later.
                // We'll extract the binary path from the symbol string.
                std::string func_info;
                bool found_function = false;

                // First, try to extract the binary path from the symbol string
                // Format: /path/to/binary(function+offset) [0xaddr]
                size_t start = symbol_str.find('(');
                size_t end = symbol_str.find(')', start);

                if (start != std::string::npos && end != std::string::npos && end > start + 1) {
                    // Extract everything between ( and )
                    std::string mangled_with_offset = symbol_str.substr(start + 1, end - start - 1);

                    // Remove offset if present (+0x123)
                    size_t plus_pos = mangled_with_offset.find('+');
                    std::string mangled =
                        (plus_pos != std::string::npos) ? mangled_with_offset.substr(0, plus_pos) : mangled_with_offset;

                    // Check if this is just an offset (e.g., "+0x1234") without a function name
                    bool is_just_offset = (!mangled.empty() && mangled[0] == '+');

                    // Try to demangle if it looks like a C++ mangled name
                    if (!is_just_offset && !mangled.empty() && mangled[0] == '_' && mangled[1] == 'Z') {
                        int status = 0;
                        char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);

                        if (status == 0 && demangled) {
                            func_info = std::string(demangled);
                            free(demangled);

                            // Simplify function name - remove template parameters and namespace details
                            size_t template_pos = func_info.find('<');
                            if (template_pos != std::string::npos) {
                                func_info = func_info.substr(0, template_pos);
                            }

                            // Extract just the function name from namespace::class::function
                            size_t last_colon = func_info.rfind("::");
                            if (last_colon != std::string::npos && last_colon + 2 < func_info.length()) {
                                // Keep class::function or just function
                                size_t prev_colon = func_info.rfind("::", last_colon - 1);
                                if (prev_colon != std::string::npos) {
                                    func_info = func_info.substr(prev_colon + 2);
                                }
                            }

                            found_function = true;
                        }
                    } else if (!is_just_offset && !mangled.empty()) {
                        // We found a recognizable function name (like "main" or C++ mangled name)
                        // Try to demangle it even if it doesn't start with _Z (might be partial mangling)
                        int status = 0;
                        char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);

                        if (status == 0 && demangled) {
                            // Successfully demangled
                            func_info = std::string(demangled);
                            free(demangled);

                            // Simplify: just keep the function name, strip parameters and templates
                            size_t paren = func_info.find('(');
                            if (paren != std::string::npos) {
                                func_info = func_info.substr(0, paren);
                            }
                            size_t template_pos = func_info.find('<');
                            if (template_pos != std::string::npos) {
                                func_info = func_info.substr(0, template_pos);
                            }

                            found_function = true;
                        } else {
                            // Not a mangled name, use as-is (e.g., "main", "perform_final_add")
                            func_info = mangled;
                            found_function = true;
                        }
                    }
                    // If is_just_offset is true, found_function stays false and we'll use the binary path below
                }

                // If we extracted a function name, format as [binary(function+offset)]
                // If we couldn't extract a function name, format as [binary(+offset)]
                if (!found_function) {
                    size_t path_end = symbol_str.find('[');
                    if (path_end != std::string::npos) {
                        std::string path = symbol_str.substr(0, path_end);
                        // Trim whitespace
                        path.erase(path.find_last_not_of(" \t") + 1);

                        // Extract just the binary path (before the parenthesis)
                        size_t paren_pos = path.find('(');
                        std::string binary_part = (paren_pos != std::string::npos) ? path.substr(0, paren_pos) : path;

                        // Make the binary path relative to current working directory for addr2line
                        std::filesystem::path binary_path(binary_part);
                        std::string relative_path;
                        try {
                            std::filesystem::path cwd = std::filesystem::current_path();
                            if (binary_path.is_absolute()) {
                                relative_path = std::filesystem::relative(binary_path, cwd).string();
                            } else {
                                relative_path = binary_path.string();
                            }
                        } catch (...) {
                            // If relative path fails, just use filename
                            relative_path = binary_path.filename().string();
                        }

                        // Use the actual address from buffer[i] and convert to file offset
                        // For PIE executables and shared libraries, we need to subtract the base load address
                        Dl_info dl_info;
                        std::stringstream addr_stream;
                        if (dladdr(buffer[i], &dl_info) && dl_info.dli_fbase != nullptr) {
                            // Calculate offset from module base address
                            uintptr_t offset =
                                reinterpret_cast<uintptr_t>(buffer[i]) - reinterpret_cast<uintptr_t>(dl_info.dli_fbase);
                            addr_stream << "(+0x" << std::hex << offset << ")";
                        } else {
                            // Fallback to absolute address if dladdr fails
                            addr_stream << "(+0x" << std::hex << reinterpret_cast<uintptr_t>(buffer[i]) << ")";
                        }
                        std::string addr_str = addr_stream.str();

                        func_info = "[" + relative_path + addr_str + "]";
                    } else {
                        continue;  // Skip this frame if we can't extract any info
                    }
                }

                // Append the function info to callstack with frame numbering
                // If we have a function name but also want address info for resolution
                if (!func_info.empty() && found_function) {
                    // We have a function name - also include binary+offset for addr2line resolution
                    size_t path_end = symbol_str.find('[');
                    if (path_end != std::string::npos) {
                        std::string path = symbol_str.substr(0, path_end);
                        path.erase(path.find_last_not_of(" \t") + 1);

                        size_t paren_pos = path.find('(');
                        std::string binary_part = (paren_pos != std::string::npos) ? path.substr(0, paren_pos) : path;

                        std::filesystem::path binary_path(binary_part);
                        std::string relative_path;
                        try {
                            std::filesystem::path cwd = std::filesystem::current_path();
                            if (binary_path.is_absolute()) {
                                relative_path = std::filesystem::relative(binary_path, cwd).string();
                            } else {
                                relative_path = binary_path.string();
                            }
                        } catch (...) {
                            relative_path = binary_path.filename().string();
                        }

                        // Format as: function [binary(+offset)]
                        Dl_info dl_info;
                        std::stringstream addr_stream;
                        if (dladdr(buffer[i], &dl_info) && dl_info.dli_fbase != nullptr) {
                            uintptr_t offset =
                                reinterpret_cast<uintptr_t>(buffer[i]) - reinterpret_cast<uintptr_t>(dl_info.dli_fbase);
                            addr_stream << "(+0x" << std::hex << offset << ")";
                        } else {
                            addr_stream << "(+0x" << std::hex << reinterpret_cast<uintptr_t>(buffer[i]) << ")";
                        }

                        if (valid_frames > 0) {
                            callstack << " ";
                        }
                        callstack << "#" << valid_frames << " " << func_info << " [" << relative_path
                                  << addr_stream.str() << "]";
                        valid_frames++;
                    }
                } else if (!func_info.empty()) {
                    // No function name, just address
                    if (valid_frames > 0) {
                        callstack << " ";
                    }
                    callstack << "#" << valid_frames << " " << func_info;
                    valid_frames++;
                }
            }

            free(symbols);
            std::string result = callstack.str();
            return result;
        }
    }

    return "";  // Return empty string if unable to get callstack
}

std::string get_callstack() {
    // Try Python callstack first
    std::string python_stack = get_python_callstack();
    if (!python_stack.empty()) {
        return python_stack;
    }

    // Fall back to C++ callstack
    return get_cpp_callstack();
}

}  // namespace tt::tt_metal::inspector
