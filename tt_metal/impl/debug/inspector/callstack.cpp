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
    std::string result;  // Store result to return after releasing GIL

    // Keep track of frames we need to clean up
    std::vector<PyFrameObject*> frames_to_clean;

    try {
        PyFrameObject* frame = dyn_PyEval_GetFrame();  // Borrowed reference

        if (frame != nullptr) {
            // Traverse up the stack to find the first user frame
            while (frame != nullptr) {
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

                                // Try to make path relative from tests/ or ttnn/ directory
                                std::filesystem::path filepath(filename_str);
                                std::string relative_path = filepath.filename().string();

                                auto tests_pos = filename_str.find("/tests/");
                                auto ttnn_pos = filename_str.find("/ttnn/");
                                if (tests_pos != std::string::npos) {
                                    relative_path = filename_str.substr(tests_pos + 1);
                                } else if (ttnn_pos != std::string::npos) {
                                    relative_path = filename_str.substr(ttnn_pos + 1);
                                } else {
                                    relative_path = filepath.filename().string();
                                }

                                std::stringstream ss;
                                ss << relative_path << ":" << lineno;
                                result = ss.str();

                                // Clean up the filename object
                                if (dyn_Py_DecRef) {
                                    dyn_Py_DecRef(filename_obj);
                                }

                                // Found user frame, exit
                                break;
                            }
                        }

                        // Always clean up the filename object
                        if (dyn_Py_DecRef) {
                            dyn_Py_DecRef(filename_obj);
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
    return result;
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

            // Skip the first 3 frames (this function, get_callstack, and track_operation)
            for (int i = 3; i < nptrs && i < 15 && valid_frames < 5; ++i) {
                std::string symbol_str(symbols[i]);

                // Skip internal Inspector frames
                if (symbol_str.find("Inspector::") != std::string::npos ||
                    symbol_str.find("inspector::") != std::string::npos ||
                    symbol_str.find("get_callstack") != std::string::npos ||
                    symbol_str.find("get_cpp_callstack") != std::string::npos ||
                    symbol_str.find("get_python_callstack") != std::string::npos ||
                    symbol_str.find("track_operation") != std::string::npos) {
                    continue;
                }

                // Try to parse the symbol to extract function name
                // Format varies by platform, typically:
                // ./executable(function+0x123) [0xaddr]
                // or /path/to/lib.so(_ZN...) [0xaddr]
                // or sometimes just: /path/to/lib.so [0xaddr]
                size_t start = symbol_str.find('(');
                size_t end = symbol_str.find(')', start);

                std::string func_info;
                bool found_function = false;

                if (start != std::string::npos && end != std::string::npos && end > start + 1) {
                    // Extract everything between ( and )
                    std::string mangled_with_offset = symbol_str.substr(start + 1, end - start - 1);

                    // Remove offset if present (+0x123)
                    size_t plus_pos = mangled_with_offset.find('+');
                    std::string mangled =
                        (plus_pos != std::string::npos) ? mangled_with_offset.substr(0, plus_pos) : mangled_with_offset;

                    // Try to demangle if it looks like a C++ mangled name
                    if (!mangled.empty() && mangled[0] == '_' && mangled[1] == 'Z') {
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
                    } else if (!mangled.empty()) {
                        // Use the unmangled name as-is
                        func_info = mangled;
                        found_function = true;
                    }
                }

                // If we couldn't extract a function name, try to get the binary name at least
                if (!found_function) {
                    size_t path_end = symbol_str.find('[');
                    if (path_end != std::string::npos) {
                        std::string path = symbol_str.substr(0, path_end);
                        // Trim whitespace
                        path.erase(path.find_last_not_of(" \t") + 1);

                        // Extract the binary path and offset separately
                        // Format: /path/to/binary(+0xoffset)
                        size_t paren_pos = path.find('(');
                        std::string binary_part = (paren_pos != std::string::npos) ? path.substr(0, paren_pos) : path;
                        std::string offset_part = (paren_pos != std::string::npos) ? path.substr(paren_pos) : "";

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

                        func_info = "[" + relative_path + offset_part + "]";
                    } else {
                        continue;  // Skip this frame if we can't extract any info
                    }
                }

                // Append the function info to callstack
                if (!func_info.empty()) {
                    if (valid_frames > 0) {
                        callstack << " <- ";
                    }
                    callstack << func_info;
                    valid_frames++;
                }
            }

            free(symbols);
            return callstack.str();
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
