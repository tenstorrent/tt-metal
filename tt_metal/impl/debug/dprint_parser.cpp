// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint_parser.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <string>
#include <string_view>

#include <enchantum/enchantum.hpp>
#include <enchantum/scoped.hpp>
#include "impl/data_format/blockfloat_common.hpp"
#include <tt_stl/assert.hpp>

#include "fmt/base.h"
#include "hostdevcommon/dprint_common.h"
#include "hostdevcommon/kernel_structs.h"
#include "tt_backend_api_types.hpp"

using std::setw;
using std::string;
using std::to_string;

namespace tt::tt_metal {

DPrintParser::DPrintParser(std::string line_prefix) : line_prefix_(std::move(line_prefix)) {}

// Helper function implementations (from dprint_server.cpp anonymous namespace)

inline float DPrintParser::bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t uint32_data = ((uint32_t)bfloat_val) << 16;
    float f;
    std::memcpy(&f, &uint32_data, sizeof(f));
    return f;
}

void DPrintParser::AssertSize(uint8_t sz, uint8_t expected_sz) {
    TT_ASSERT(
        sz == expected_sz,
        "DPrint token size ({}) did not match expected ({}), potential data corruption in the DPrint buffer.",
        sz,
        expected_sz);
}

void DPrintParser::ResetStream(std::ostringstream* stream) {
    stream->str("");
    stream->clear();
}

bool DPrintParser::StreamEndsWithNewlineChar(const std::ostringstream* stream) {
    const string stream_str = stream->str();
    return !stream_str.empty() && stream_str.back() == '\n';
}

void DPrintParser::PrintTileSlice(const uint8_t* ptr) {
    TileSliceHostDev<0> ts_copy{};  // Make a copy since ptr might not be properly aligned
    std::memcpy(&ts_copy, ptr, sizeof(TileSliceHostDev<0>));
    TileSliceHostDev<0>* ts = &ts_copy;
    TT_ASSERT(
        offsetof(TileSliceHostDev<0>, data) % sizeof(uint32_t) == 0,
        "TileSliceHostDev<0> data field is not properly aligned");
    const uint8_t* data = ptr + offsetof(TileSliceHostDev<0>, data);

    // Read any error codes and handle accordingly
    tt::CBIndex cb = static_cast<tt::CBIndex>(ts->cb_id);
    switch (ts->return_code) {
        case DPrintOK: break;  // Continue to print the tile slice
        case DPrintErrorBadPointer: {
            uint32_t cb_ptr_val = ts->cb_ptr;
            uint8_t count = ts->data_count;
            intermediate_stream_ << fmt::format(
                "Tried printing {}: BAD TILE POINTER (ptr={}, count={})\n",
                enchantum::scoped::to_string(cb),
                cb_ptr_val,
                count);
            return;
        }
        case DPrintErrorUnsupportedFormat: {
            tt::DataFormat data_format = static_cast<tt::DataFormat>(ts->data_format);
            intermediate_stream_ << fmt::format(
                "Tried printing {}: Unsupported data format ({})\n",
                enchantum::scoped::to_string(cb),
                data_format);
            return;
        }
        case DPrintErrorMath:
            intermediate_stream_ << "Warning: MATH core does not support TileSlice printing, omitting print...\n";
            return;
        case DPrintErrorEthernet:
            intermediate_stream_ << "Warning: Ethernet core does not support TileSlice printing, omitting print...\n";
            return;
        default:
            intermediate_stream_ << fmt::format(
                "Warning: TileSlice printing failed with unknown return code {}, omitting print...\n", ts->return_code);
            return;
    }

    // No error codes, print the TileSlice
    uint32_t i = 0;
    bool count_exceeded = false;
    for (int h = ts->slice_range.h0; h < ts->slice_range.h1; h += ts->slice_range.hs) {
        for (int w = ts->slice_range.w0; w < ts->slice_range.w1; w += ts->slice_range.ws) {
            // If the number of data specified by the SliceRange exceeds the number that was
            // saved in the print buffer (set by the MAX_COUNT template parameter in the
            // TileSlice), then break early.
            if (i >= ts->data_count) {
                count_exceeded = true;
                break;
            }
            tt::DataFormat data_format = static_cast<tt::DataFormat>(ts->data_format);
            switch (data_format) {
                case tt::DataFormat::Float16_b: {
                    const uint16_t* float16_b_ptr = reinterpret_cast<const uint16_t*>(data);
                    intermediate_stream_ << bfloat16_to_float(float16_b_ptr[i]);
                    break;
                }
                case tt::DataFormat::Float32: {
                    const float* float32_ptr = reinterpret_cast<const float*>(data);
                    intermediate_stream_ << float32_ptr[i];
                    break;
                }
                case tt::DataFormat::Bfp4_b:
                case tt::DataFormat::Bfp8_b: {
                    // Saved the exponent and data together
                    const uint16_t* data_ptr = reinterpret_cast<const uint16_t*>(data);
                    uint8_t val = (data_ptr[i] >> 8) & 0xFF;
                    uint8_t exponent = data_ptr[i] & 0xFF;
                    uint32_t bit_val = convert_bfp_to_u32(data_format, val, exponent, false);
                    intermediate_stream_ << *reinterpret_cast<float*>(&bit_val);
                    break;
                }
                case tt::DataFormat::Int8: {
                    const int8_t* data_ptr = reinterpret_cast<const int8_t*>(data);
                    intermediate_stream_ << (int)data_ptr[i];
                    break;
                }
                case tt::DataFormat::UInt8: {
                    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(data);
                    intermediate_stream_ << (unsigned int)data_ptr[i];
                    break;
                }
                case tt::DataFormat::UInt16: {
                    const uint16_t* data_ptr = reinterpret_cast<const uint16_t*>(data);
                    intermediate_stream_ << (unsigned int)data_ptr[i];
                    break;
                }
                case tt::DataFormat::Int32: {
                    const int32_t* data_ptr = reinterpret_cast<const int32_t*>(data);
                    intermediate_stream_ << (int)data_ptr[i];
                    break;
                }
                case tt::DataFormat::UInt32: {
                    const uint32_t* data_ptr = reinterpret_cast<const uint32_t*>(data);
                    intermediate_stream_ << (unsigned int)data_ptr[i];
                    break;
                }
                default: break;
            }
            if (w + ts->slice_range.ws < ts->slice_range.w1) {
                intermediate_stream_ << " ";
            }
            i++;
        }

        // Break outer loop as well if MAX COUNT exceeded, also print a message to let the user
        // know that the slice has been truncated.
        if (count_exceeded) {
            intermediate_stream_ << "<TileSlice data truncated due to exceeding max count ("
                                 << to_string(ts->data_count) << ")>\n";
            break;
        }

        if (ts->endl_rows) {
            intermediate_stream_ << "\n";
        }
    }
}

// Create a float from a given bit pattern, given the number of bits for the exponent and mantissa.
// Assumes the following order of bits in the input data:
//   [sign bit][mantissa bits][exponent bits]
float DPrintParser::make_float(uint8_t exp_bit_count, uint8_t mantissa_bit_count, uint32_t data) {
    int sign = (data >> (exp_bit_count + mantissa_bit_count)) & 0x1;
    const int exp_mask = (1 << (exp_bit_count)) - 1;
    int exp_bias = (1 << (exp_bit_count - 1)) - 1;
    int exp_val = (data & exp_mask) - exp_bias;
    const int mantissa_mask = ((1 << mantissa_bit_count) - 1) << exp_bit_count;
    int mantissa_val = (data & mantissa_mask) >> exp_bit_count;
    float result = 1.0 + ((float)mantissa_val / (float)(1 << mantissa_bit_count));
    result = result * pow(2, exp_val);
    if (sign) {
        result = -result;
    }
    return result;
}

// Prints a given datum in the array, given the data_format
void DPrintParser::PrintTensixRegisterData(int setwidth, uint32_t datum, uint16_t data_format) {
    switch (data_format) {
        case static_cast<std::uint8_t>(tt::DataFormat::Float16):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp8):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp4):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp2):
        case static_cast<std::uint8_t>(tt::DataFormat::Lf8):
            intermediate_stream_ << setw(setwidth) << make_float(5, 10, datum & 0xffff) << " ";
            intermediate_stream_ << setw(setwidth) << make_float(5, 10, (datum >> 16) & 0xffff) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp8_b):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp4_b):
        case static_cast<std::uint8_t>(tt::DataFormat::Bfp2_b):
        case static_cast<std::uint8_t>(tt::DataFormat::Float16_b):
            intermediate_stream_ << setw(setwidth) << make_float(8, 7, datum & 0xffff) << " ";
            intermediate_stream_ << setw(setwidth) << make_float(8, 7, (datum >> 16) & 0xffff) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Tf32):
            intermediate_stream_ << setw(setwidth) << make_float(8, 10, datum) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Float32): {
            float value;
            memcpy(&value, &datum, sizeof(float));
            intermediate_stream_ << setw(setwidth) << value << " ";
        } break;
        case static_cast<std::uint8_t>(tt::DataFormat::UInt32):
            intermediate_stream_ << setw(setwidth) << datum << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::UInt16):
            intermediate_stream_ << setw(setwidth) << (datum & 0xffff) << " ";
            intermediate_stream_ << setw(setwidth) << (datum >> 16) << " ";
            break;
        case static_cast<std::uint8_t>(tt::DataFormat::Int32):
            intermediate_stream_ << setw(setwidth) << static_cast<int32_t>(datum) << " ";
            break;
        default: intermediate_stream_ << "Unknown data format " << data_format << " "; break;
    }
}

// Prints a typed uint32 array given the number of elements including the type.
// If force_element_type is set to a valid type, it is assumed that the type is not included in the
// data array, and the type is forced to be the given type.
void DPrintParser::PrintTypedUint32Array(
    int setwidth, uint32_t raw_element_count, const uint32_t* data, TypedU32_ARRAY_Format force_array_type) {
    uint16_t array_type = data[raw_element_count - 1] >> 16;
    uint16_t array_subtype = data[raw_element_count - 1] & 0xffff;

    raw_element_count = (force_array_type == TypedU32_ARRAY_Format_INVALID) ? raw_element_count : raw_element_count + 1;

    for (uint32_t i = 0; i < raw_element_count - 1; i++) {
        switch (array_type) {
            case TypedU32_ARRAY_Format_Raw: intermediate_stream_ << std::hex << "0x" << data[i] << " "; break;
            case TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type:
                PrintTensixRegisterData(setwidth, data[i], array_subtype);
                break;
            default: intermediate_stream_ << "Unknown type " << array_type; break;
        }
    }
}

// Helper to collect a completed line with prefix
std::string DPrintParser::get_completed_line() {
    std::string line;
    if (!line_prefix_.empty()) {
        line += line_prefix_;
    }
    if (!intermediate_stream_.str().empty()) {
        line += intermediate_stream_.str();
    }
    ResetStream(&intermediate_stream_);
    return line;
}

DPrintParser::ParseResult DPrintParser::parse(const uint8_t* data, size_t len) {
    ParseResult result;
    result.bytes_consumed = 0;

    // Parse the input codes
    size_t pos = 0;
    while (pos < len) {
        DPrintTypeID code = static_cast<DPrintTypeID>(data[pos++]);
        TT_ASSERT(pos <= len);
        uint8_t sz = data[pos++];
        TT_ASSERT(pos <= len);
        const uint8_t* ptr = data + pos;

        // Possible to break before pos == len due to waiting on another core's raise.
        bool break_due_to_wait = false;

        // we are sharing the same output file between debug print threads for multiple cores
        switch (code) {
            case DPrintCSTR:  // const char*
            {
                // null terminating char was included in size and should be present in the buffer
                const char* cptr = reinterpret_cast<const char*>(ptr);
                const size_t cptr_len = strnlen(cptr, len - 2);
                if (cptr_len == len - 2) {
                    intermediate_stream_ << "STRING BUFFER OVERFLOW DETECTED\n";
                    result.completed_lines.push_back(get_completed_line());
                } else {
                    // if we come across a newline char, we should transfer the data up to the newline to the output
                    // stream and flush it
                    const char* newline_pos = strchr(cptr, '\n');
                    bool contains_newline = newline_pos != nullptr;
                    while (contains_newline) {
                        const char* pos_after_newline = newline_pos + 1;
                        const uint32_t substr_len = pos_after_newline - cptr;

                        // strchr returns nullptr if it encounters a null terminator,
                        // so we can guarantee that this is valid data since it was
                        // already checked. We don't need to append a '\0' because
                        // the stream operator only takes upto '\0' when passed
                        // a char* (wrt the previous impl)
                        const std::string_view substr_upto_newline(cptr, substr_len);

                        intermediate_stream_ << substr_upto_newline;
                        result.completed_lines.push_back(get_completed_line());
                        cptr = pos_after_newline;
                        newline_pos = strchr(cptr, '\n');
                        contains_newline = newline_pos != nullptr;
                    }
                    intermediate_stream_ << cptr;
                }
                AssertSize(sz, cptr_len + 1);
                break;
            }
            case DPrintTILESLICE: PrintTileSlice(ptr); break;

            case DPrintENDL:
                if (prev_type_ != DPrintTILESLICE || !StreamEndsWithNewlineChar(&intermediate_stream_)) {
                    intermediate_stream_ << '\n';
                }
                result.completed_lines.push_back(get_completed_line());
                AssertSize(sz, 1);
                break;
            case DPrintSETW: {
                char val = ptr[0];
                intermediate_stream_ << setw(val);
                most_recent_setw_ = val;
                AssertSize(sz, 1);
                break;
            }
            case DPrintSETPRECISION:
                intermediate_stream_ << std::setprecision(*ptr);
                AssertSize(sz, 1);
                break;
            case DPrintFIXED:
                intermediate_stream_ << std::fixed;
                AssertSize(sz, 1);
                break;
            case DPrintDEFAULTFLOAT:
                intermediate_stream_ << std::defaultfloat;
                AssertSize(sz, 1);
                break;
            case DPrintHEX:
                intermediate_stream_ << std::hex;
                AssertSize(sz, 1);
                break;
            case DPrintOCT:
                intermediate_stream_ << std::oct;
                AssertSize(sz, 1);
                break;
            case DPrintDEC:
                intermediate_stream_ << std::dec;
                AssertSize(sz, 1);
                break;
            case DPrintUINT8:
                // iostream default uint8_t printing is as char, not an int
                intermediate_stream_ << *reinterpret_cast<const uint8_t*>(ptr);
                AssertSize(sz, 1);
                break;
            case DPrintUINT16: {
                uint16_t value;
                memcpy(&value, ptr, sizeof(uint16_t));
                intermediate_stream_ << value;
                AssertSize(sz, 2);
            } break;
            case DPrintUINT32: {
                uint32_t value;
                memcpy(&value, ptr, sizeof(uint32_t));
                intermediate_stream_ << value;
                AssertSize(sz, 4);
            } break;
            case DPrintUINT64: {
                uint64_t value;
                memcpy(&value, ptr, sizeof(uint64_t));
                intermediate_stream_ << value;
                AssertSize(sz, 8);
            } break;
            case DPrintINT8: {
                int8_t value;
                memcpy(&value, ptr, sizeof(int8_t));
                intermediate_stream_ << (int)value;  // Cast to int to ensure it prints as a number, not a char
                AssertSize(sz, 1);
            } break;
            case DPrintINT16: {
                int16_t value;
                memcpy(&value, ptr, sizeof(int16_t));
                intermediate_stream_ << value;
                AssertSize(sz, 2);
            } break;
            case DPrintINT32: {
                int32_t value;
                memcpy(&value, ptr, sizeof(int32_t));
                intermediate_stream_ << value;
                AssertSize(sz, 4);
            } break;
            case DPrintINT64: {
                int64_t value;
                memcpy(&value, ptr, sizeof(int64_t));
                intermediate_stream_ << value;
                AssertSize(sz, 8);
            } break;
            case DPrintFLOAT32: {
                float value;
                memcpy(&value, ptr, sizeof(float));
                intermediate_stream_ << value;
                AssertSize(sz, 4);
            } break;
            case DPrintBFLOAT16: {
                uint16_t rawValue;
                memcpy(&rawValue, ptr, sizeof(uint16_t));
                float value = bfloat16_to_float(rawValue);
                intermediate_stream_ << value;
                AssertSize(sz, 2);
            } break;
            case DPrintCHAR:
                intermediate_stream_ << *reinterpret_cast<const char*>(ptr);
                AssertSize(sz, 1);
                break;
            case DPrintU32_ARRAY:
                PrintTypedUint32Array(
                    most_recent_setw_, sz / 4, reinterpret_cast<const uint32_t*>(ptr), TypedU32_ARRAY_Format_Raw);
                break;
            case DPrintTYPED_U32_ARRAY:
                PrintTypedUint32Array(most_recent_setw_, sz / 4, reinterpret_cast<const uint32_t*>(ptr));
                break;
            default: TT_THROW("Unexpected debug print type pos {:#x} len {:#x} code {}", pos, len, (uint32_t)code);
        }

        prev_type_ = code;

        pos += sz;  // parse the payload size
        TT_ASSERT(pos <= len);

        // Break due to wait (we'll get the rest of the print buffer after the raise).
        if (break_due_to_wait) {
            break;
        }
    }

    result.bytes_consumed = pos;
    return result;
}

std::string DPrintParser::flush() { return get_completed_line(); }

}  // namespace tt::tt_metal
