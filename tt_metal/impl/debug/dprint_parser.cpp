// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint_parser.hpp"

#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <string>
#include <string_view>

#include <enchantum/enchantum.hpp>
#include <enchantum/scoped.hpp>
#include "device/device_impl.hpp"
#include "impl/data_format/blockfloat_common.hpp"
#include <tt_stl/assert.hpp>

#include "fmt/base.h"
#include "hostdevcommon/dprint_common.h"
#include "hostdevcommon/kernel_structs.h"
#include "tt_backend_api_types.hpp"

#include "dwarf.h"
#include "libdwarf.h"

using std::setw;
using std::string;
using std::to_string;
using namespace std::literals;

namespace tt::tt_metal {

DPrintParser::DPrintParser(std::string line_prefix) : line_prefix_(std::move(line_prefix)) {}

// Helper function implementations (from dprint_server.cpp anonymous namespace)

inline float bfloat16_to_float(uint16_t bfloat_val) {
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
            case DPrintUINT8: {
                uint8_t value;
                memcpy(&value, ptr, sizeof(uint8_t));
                intermediate_stream_ << static_cast<uint32_t>(value);  // uint8_t is unsigned char; widen to print as number
                AssertSize(sz, 1);
            } break;
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

std::map<std::string, std::weak_ptr<DevicePrintParser>> DevicePrintParser::parser_cache;

DevicePrintParser::DevicePrintParser(const std::string& elf_path) : elf_path(elf_path) {
    try {
        elf_file.ReadImage(elf_path);
        format_strings_info_bytes =
            elf_file.GetSectionContents(".device_print_strings_info", format_strings_info_address);
        format_strings_bytes = elf_file.GetSectionContents(".device_print_strings", format_strings_address);
        string_info_ptr = reinterpret_cast<DevicePrintStringInfo*>(format_strings_info_bytes.data());
        string_info_size = format_strings_info_bytes.size() / sizeof(DevicePrintStringInfo);
        parsed_string_info.resize(string_info_size);
    } catch (...) {
        // Failed to load ELF file
        log_warning(tt::LogMetal, "Failed to load ELF file {}", elf_path);
    }
}

struct DevicePrintParserDeleter {
    void operator()(DevicePrintParser* parser) const {
        DevicePrintParser::parser_cache.erase(parser->elf_path);
        delete parser;
    }
};

std::shared_ptr<DevicePrintParser> DevicePrintParser::get_parser_for_elf(const std::string& elf_path) {
    auto cached_parser_it = parser_cache.find(elf_path);
    if (cached_parser_it != parser_cache.end()) {
        if (auto cached_parser = cached_parser_it->second.lock()) {
            return cached_parser;
        }
    }
    std::shared_ptr<DevicePrintParser> new_parser(new DevicePrintParser(elf_path), DevicePrintParserDeleter());
    parser_cache[elf_path] = new_parser;
    return new_parser;
}

std::string_view DevicePrintParser::format_message(
    uint32_t info_id, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer) {
    auto* string_info = get_string_info(info_id);
    if (string_info == nullptr) {
        return {};
    }
    return format_message(*string_info, payload_bytes, buffer);
}

DevicePrintParser::ParsedStringInfo* DevicePrintParser::get_string_info(uint32_t info_id) {
    if (info_id >= string_info_size) {
        return nullptr;
    }
    auto& parsed_info = parsed_string_info[info_id];
    if (parsed_info.format_string.empty()) {
        // This entry has not been parsed yet, so parse it now and cache the result.
        const DevicePrintStringInfo& info = string_info_ptr[info_id];
        if (info.format_string_ptr >= format_strings_address &&
            info.format_string_ptr < format_strings_address + format_strings_bytes.size()) {
            const char* format_string = reinterpret_cast<const char*>(
                format_strings_bytes.data() + (info.format_string_ptr - format_strings_address));
            parsed_info.format_string = format_string;
            std::tie(parsed_info.plain_text_parts, parsed_info.placeholders) = parse_format_string(format_string);
            if (!parsed_info.placeholders.empty()) {
                uint32_t max_arg_id = 0;
                for (const auto& placeholder : parsed_info.placeholders) {
                    max_arg_id = std::max(max_arg_id, placeholder.arg_id);
                }
                parsed_info.argument_types.resize(max_arg_id + 1);
                for (auto& placeholder : parsed_info.placeholders) {
                    // For long types (type_id == '/'), the serialization uses the base type
                    char serialization_type = placeholder.type_id;
                    if (placeholder.type_id == '/' && placeholder.enum_base_type_id != 0) {
                        // '/e' enum type: serialize as the underlying integer type
                        serialization_type = placeholder.enum_base_type_id;
                        placeholder.enum_info = get_enum_info(placeholder.enum_type_name);
                    }
                    parsed_info.argument_types[placeholder.arg_id] = serialization_type;
                    parsed_info.arguments_size += get_argument_size_from_type_id(serialization_type);
                }
            }
        }
        if (info.file >= format_strings_address && info.file < format_strings_address + format_strings_bytes.size()) {
            const char* file =
                reinterpret_cast<const char*>(format_strings_bytes.data() + (info.file - format_strings_address));
            parsed_info.file = file;
        }
        parsed_info.line = info.line;
    }
    return &parsed_info;
}

template <typename T>
T read_value_from_payload(std::span<const std::byte> payload_bytes, std::size_t& offset) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset + sizeof(T) > payload_bytes.size()) {
        TT_THROW("Payload does not contain enough bytes to read type");
    }
    T value;
    std::memcpy(&value, payload_bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

std::size_t DevicePrintParser::get_argument_size_from_type_id(char type_id) {
    static std::byte empty_bytes[32];
    std::size_t offset = 0;
    read_argument_from_payload(type_id, std::span<const std::byte>(empty_bytes), offset);
    return offset;
}

void DevicePrintParser::read_arguments_from_payload(
    std::span<char> argument_types, std::span<const std::byte> payload_bytes, std::vector<ArgumentValue>& arguments) {
    std::size_t payload_offset = 0;

    arguments.clear();
    arguments.reserve(argument_types.size());
    for (char argument_type : argument_types) {
        arguments.push_back(read_argument_from_payload(argument_type, payload_bytes, payload_offset));
    }
}

std::pair<std::vector<std::string>, std::vector<DevicePrintParser::FormatPlaceholderInfo>>
DevicePrintParser::parse_format_string(std::string_view format_str) {
    std::vector<std::string> plain_text_parts;
    std::vector<FormatPlaceholderInfo> placeholders;
    fmt::memory_buffer current_text;
    for (size_t i = 0; i < format_str.size(); i++) {
        if (format_str[i] == '{' && i + 1 < format_str.size() && format_str[i + 1] == '{') {
            // Escaped '{', add a single '{' to the result and skip the next character.
            current_text.push_back('{');
            i++;
            continue;
        }
        if (format_str[i] == '}' && i + 1 < format_str.size() && format_str[i + 1] == '}') {
            // Escaped '}', add a single '}' to the result and skip the next character.
            current_text.push_back('}');
            i++;
            continue;
        }
        if (format_str[i] == '{') {
            auto placeholder = parse_placeholder(format_str, i);
            if (!placeholder) {
                TT_THROW("Invalid format string: failed to parse placeholder at position {}", i);
            }
            placeholders.push_back(*placeholder);
            plain_text_parts.push_back(std::string(current_text.data(), current_text.size()));
            current_text.clear();
            i--;  // Step back so that the main loop can correctly identify the end of the placeholder
        } else {
            // Regular character, add it to the result.
            current_text.push_back(format_str[i]);
            continue;
        }
    }
    plain_text_parts.push_back(std::string(current_text.data(), current_text.size()));
    return {std::move(plain_text_parts), std::move(placeholders)};
}

std::optional<DevicePrintParser::FormatPlaceholderInfo> DevicePrintParser::parse_placeholder(
    std::string_view format_str, std::size_t& pos) {
    if (pos >= format_str.size() || format_str[pos] != '{') {
        return std::nullopt;
    }

    // Start of a placeholder. Read until the closing '}' to extract the placeholder content.
    pos++;  // Skip '{'

    // We are trying to mimic fmtlib format specifiers here, but device already changed it a bit:
    // replacement_field ::= "{" arg_id "," type_id [":" format_spec] "}"
    // arg_id            ::= integer
    // integer           ::= digit+
    // digit             ::= "0"..."9"
    // type_id           ::= short_type | long_type
    // short_type        ::= "a"..."z" | "A"..."Z"         (single character, e.g. 'I' for uint32_t)
    // long_type         ::= "/" sub_type "_" extra_info   ('/' signals a multi-character type descriptor)
    // sub_type          ::= "e"                           (regular enum)
    //                     | "E"                           (flag/bitmask enum with operator|)
    // extra_info        ::= short_type "_" enum_name      (e.g. "I_test::deep::Enum1")
    // But we don't support using identifiers to reduce kernel size, only integers for arg_id.

    // Regarding format_spec:
    // format_spec ::= [[fill]align][sign]["#"]["0"][width]["." precision]["L"][type]
    // fill        ::= <a character other than '{' or '}'>
    // align       ::= "<" | ">" | "^"
    // sign        ::= "+" | "-" | " "
    // width       ::= integer | "{" [arg_id] "}"
    // precision   ::= integer | "{" [arg_id] "}"
    // type        ::= "a" | "A" | "b" | "B" | "c" | "d" | "e" | "E" | "f" | "F" |
    //                 "g" | "G" | "o" | "p" | "s" | "x" | "X" | "?"
    // We don't support using arg_id for width/precision.

    // As everything is verified during kernel compile time, we can parse format_spec just by reading until the closing
    // '}' without needing to fully understand it on the host side.
    uint32_t arg_id = 0;

    // arg_id parsing
    if (!std::isdigit(format_str[pos])) {
        return std::nullopt;
    }
    while (pos < format_str.size() && std::isdigit(format_str[pos])) {
        arg_id = arg_id * 10 + (format_str[pos] - '0');
        pos++;
    }

    // Read type_id (the character after arg_id and ',')
    if (pos >= format_str.size() || format_str[pos] != ',') {
        return std::nullopt;
    }
    pos++;  // Skip ','
    char type_id = format_str[pos++];

    // Check for long type marker: '/' followed by sub-type character.
    // Supported: /e_<base_type>_<enum_name> (regular enum), /E_<base_type>_<enum_name> (flag enum)
    if (type_id == '/' && pos + 3 < format_str.size() && (format_str[pos] == 'e' || format_str[pos] == 'E') &&
        format_str[pos + 1] == '_') {
        bool is_flag_enum = (format_str[pos] == 'E');
        pos += 2;  // Skip 'e_' or 'E_'
        char base_type = format_str[pos++];
        if (pos < format_str.size() && format_str[pos] == '_') {
            pos++;  // Skip second '_'
        }
        // Read enum type name until format spec separator or '}'.
        // A single ':' not followed by ':' marks the start of a format spec.
        // '::' is a C++ namespace separator and is part of the enum type name.
        std::size_t name_start = pos;
        while (pos < format_str.size() && format_str[pos] != '}') {
            if (format_str[pos] == ':' && (pos + 1 >= format_str.size() || format_str[pos + 1] != ':')) {
                break;  // Single ':' = format spec separator
            }
            if (format_str[pos] == ':' && pos + 1 < format_str.size() && format_str[pos + 1] == ':') {
                pos += 2;  // Skip '::' as part of the name
                continue;
            }
            pos++;
        }
        std::string_view enum_type_name = format_str.substr(name_start, pos - name_start);

        // Parse optional format spec (may contain '#' for full name)
        bool use_full_name = false;
        std::size_t format_spec_start = pos;
        while (pos < format_str.size() && format_str[pos] != '}') {
            if (format_str[pos] == '#') {
                use_full_name = true;
            }
            pos++;
        }
        auto format_spec = format_str.substr(format_spec_start, pos - format_spec_start);
        pos++;  // Skip '}'

        // Build fmt_format for the enum string, stripping '#' which is consumed as enum_use_full_name.
        std::string fmt_format = "{0";
        for (char c : format_spec) {
            if (c != '#') {
                fmt_format += c;
            }
        }
        fmt_format += '}';

        FormatPlaceholderInfo info;
        info.arg_id = arg_id;
        info.type_id = '/';  // Mark as long type (enum)
        info.format_spec = format_spec;
        info.fmt_format = std::move(fmt_format);
        info.enum_type_name = enum_type_name;
        info.enum_base_type_id = base_type;
        info.enum_is_flag = is_flag_enum;
        info.enum_use_full_name = use_full_name;
        return info;
    }

    uint32_t format_spec_start = pos;
    while (pos < format_str.size() && format_str[pos] != '}') {
        pos++;
    }
    pos++;  // Skip '}'
    auto format_spec = format_str.substr(format_spec_start, pos - format_spec_start - 1);

    FormatPlaceholderInfo info;
    info.arg_id = arg_id;
    info.type_id = type_id;
    info.format_spec = format_spec;
    info.fmt_format = "{0" + std::string(format_spec) + "}";
    return info;
}

std::string_view DevicePrintParser::format_message(
    ParsedStringInfo& string_info, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer) {
    // Iterate over format_str and replace {} with format of payload values.
    buffer.buffer.clear();
    auto format_str = string_info.format_string;
    if (string_info.arguments_size > payload_bytes.size()) {
        log_warning(
            tt::LogMetal,
            "Payload size {} is smaller than expected arguments size {} for format string '{}'",
            payload_bytes.size(),
            string_info.arguments_size,
            format_str);
        return {};
    }
    read_arguments_from_payload(string_info.argument_types, payload_bytes, buffer.argument_values);

    for (size_t i = 0; i < string_info.placeholders.size(); i++) {
        // Append prefix plain text part before the placeholder
        auto& plain_text_part = string_info.plain_text_parts[i];
        buffer.buffer.append(plain_text_part.data(), plain_text_part.data() + plain_text_part.size());

        // Append the formatted argument for the placeholder
        auto& placeholder = string_info.placeholders[i];

        // Do the actual formatting of the argument
        auto format = fmt::runtime(placeholder.fmt_format);

        switch (placeholder.type_id) {
            case 'b':  // int8_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<int8_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'B':  // uint8_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<uint8_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'h':  // int16_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<int16_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'H':  // uint16_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<uint16_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'i':  // int32_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<int32_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'I':  // uint32_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<uint32_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'q':  // int64_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<int64_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'Q':  // uint64_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<uint64_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'f':  // float
            case 'e':  // bf4_t, but stored as float
            case 'E':  // bf8_t, but stored as float
            case 'w':  // bf16_t, but stored as float
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<float>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'd':  // double
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<double>(buffer.argument_values[placeholder.arg_id]));
                break;
            case '?':  // bool
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<bool>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 't':  // TileSliceDynamic
            {
                const auto& tile_slice = std::get<TileSliceDynamic>(buffer.argument_values[placeholder.arg_id]);

                // Read any error codes and handle accordingly
                tt::CBIndex cb = static_cast<tt::CBIndex>(tile_slice.header.cb_id);
                switch (tile_slice.header.return_code) {
                    case DPrintOK: break;  // Continue to print the tile slice
                    case DPrintErrorBadPointer: {
                        uint32_t cb_ptr_val = tile_slice.header.cb_ptr;
                        uint8_t count = tile_slice.header.data_count;
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Tried printing {}: BAD TILE POINTER (ptr={}, count={})\n",
                            enchantum::scoped::to_string(cb),
                            cb_ptr_val,
                            count);
                        continue;
                    }
                    case DPrintErrorUnsupportedFormat: {
                        tt::DataFormat data_format = static_cast<tt::DataFormat>(tile_slice.header.data_format);
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Tried printing {}: Unsupported data format ({})\n",
                            enchantum::scoped::to_string(cb),
                            data_format);
                        continue;
                    }
                    case DPrintErrorMath:
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Warning: MATH core does not support TileSlice printing, omitting print...\n");
                        continue;
                    case DPrintErrorEthernet:
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Warning: Ethernet core does not support TileSlice printing, omitting print...\n");
                        continue;
                    default:
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Warning: TileSlice printing failed with unknown return code {}, omitting print...\n",
                            tile_slice.header.return_code);
                        continue;
                }

                // No error codes, print the TileSlice
                const uint8_t* data = tile_slice.data.data();
                uint32_t i = 0;
                bool count_exceeded = false;
                for (int h = tile_slice.header.slice_range.h0; h < tile_slice.header.slice_range.h1;
                     h += tile_slice.header.slice_range.hs) {
                    for (int w = tile_slice.header.slice_range.w0; w < tile_slice.header.slice_range.w1;
                         w += tile_slice.header.slice_range.ws) {
                        // If the number of data specified by the SliceRange exceeds the number that was
                        // saved in the print buffer (set by the MAX_COUNT template parameter in the
                        // TileSlice), then break early.
                        if (i >= tile_slice.header.data_count) {
                            count_exceeded = true;
                            break;
                        }
                        tt::DataFormat data_format = static_cast<tt::DataFormat>(tile_slice.header.data_format);
                        switch (data_format) {
                            case tt::DataFormat::Float16_b: {
                                const uint16_t* float16_b_ptr = reinterpret_cast<const uint16_t*>(data);
                                fmt::format_to(
                                    std::back_inserter(buffer.buffer), format, bfloat16_to_float(float16_b_ptr[i]));
                                break;
                            }
                            case tt::DataFormat::Float32: {
                                const float* float32_ptr = reinterpret_cast<const float*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, float32_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::Bfp4_b:
                            case tt::DataFormat::Bfp8_b: {
                                // Saved the exponent and data together
                                const uint16_t* data_ptr = reinterpret_cast<const uint16_t*>(data);
                                uint8_t val = (data_ptr[i] >> 8) & 0xFF;
                                uint8_t exponent = data_ptr[i] & 0xFF;
                                uint32_t bit_val = convert_bfp_to_u32(data_format, val, exponent, false);
                                fmt::format_to(
                                    std::back_inserter(buffer.buffer), format, *reinterpret_cast<float*>(&bit_val));
                                break;
                            }
                            case tt::DataFormat::Int8: {
                                const int8_t* data_ptr = reinterpret_cast<const int8_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (int)data_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::UInt8: {
                                const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (unsigned int)data_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::UInt16: {
                                const uint16_t* data_ptr = reinterpret_cast<const uint16_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (unsigned int)data_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::Int32: {
                                const int32_t* data_ptr = reinterpret_cast<const int32_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (int)data_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::UInt32: {
                                const uint32_t* data_ptr = reinterpret_cast<const uint32_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (unsigned int)data_ptr[i]);
                                break;
                            }
                            default: break;
                        }
                        if (w + tile_slice.header.slice_range.ws < tile_slice.header.slice_range.w1) {
                            buffer.buffer.append(" "sv);
                        }
                        i++;
                    }

                    // Break outer loop as well if MAX COUNT exceeded, also print a message to let the user
                    // know that the slice has been truncated.
                    if (count_exceeded) {
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "<TileSlice data truncated due to exceeding max count ({})>\n",
                            tile_slice.header.data_count);
                        break;
                    }

                    if (tile_slice.header.endl_rows) {
                        buffer.buffer.append("\n"sv);
                    }
                }

                break;
            }
            case '/':  // Long type: '/' followed by sub-type character
            {
                TT_ASSERT(placeholder.enum_base_type_id != 0, "Unsupported long type in format placeholder");
                // Currently only '/e' (enum) and /E (flag enum) are supported
                // Get the integer value from the argument (using the base type)
                int64_t int_val = 0;
                auto& arg = buffer.argument_values[placeholder.arg_id];
                std::visit(
                    [&int_val](auto&& v) {
                        using V = std::decay_t<decltype(v)>;
                        if constexpr (std::is_integral_v<V>) {
                            int_val = static_cast<int64_t>(v);
                        }
                    },
                    arg);

                // Build the enum string representation, then format it with the user's format spec.
                fmt::memory_buffer enum_str;
                auto* ei = placeholder.enum_info;

                // Append "TypeName::" prefix when full name is requested
                auto append_type_prefix = [&]() {
                    if (placeholder.enum_use_full_name) {
                        enum_str.append(placeholder.enum_type_name);
                        enum_str.append("::"sv);
                    }
                };

                // Format an unrecognized value as (TypeName)integer
                auto append_raw_value = [&](int64_t val) {
                    fmt::format_to(std::back_inserter(enum_str), "({}){}", placeholder.enum_type_name, val);
                };

                if (!ei || ei->enumerators.empty()) {
                    // No DWARF info available
                    append_raw_value(int_val);
                } else if (placeholder.enum_is_flag && int_val != 0) {
                    // Print bitfield: Flag1 | Flag3 or BitEnum::Flag1 | BitEnum::Flag3
                    bool first = true;
                    int64_t remaining = int_val;
                    for (auto& [eval, ename] : ei->enumerators) {
                        if (eval != 0 && (remaining & eval) == eval) {
                            if (!first) {
                                enum_str.append(" | "sv);
                            }
                            append_type_prefix();
                            enum_str.append(std::string_view(ename));
                            remaining &= ~eval;
                            first = false;
                        }
                    }
                    if (remaining != 0) {
                        if (!first) {
                            enum_str.append(" | "sv);
                        }
                        append_raw_value(remaining);
                    }
                } else {
                    // Non-bitfield: find exact match
                    const std::string* found_name = nullptr;
                    for (auto& [eval, ename] : ei->enumerators) {
                        if (eval == int_val) {
                            found_name = &ename;
                            break;  // Use first match
                        }
                    }
                    if (found_name) {
                        append_type_prefix();
                        enum_str.append(std::string_view(*found_name));
                    } else {
                        append_raw_value(int_val);
                    }
                }

                // Apply user's format spec (alignment, width, fill, etc.)
                auto enum_sv = std::string_view(enum_str.data(), enum_str.size());
                fmt::format_to(std::back_inserter(buffer.buffer), fmt::runtime(placeholder.fmt_format), enum_sv);
                break;
            }
            default: TT_THROW("Unsupported type_id in format placeholder (format_message): {}", placeholder.type_id);
        }
    }
    auto& plain_text_part = string_info.plain_text_parts[string_info.placeholders.size()];
    buffer.buffer.append(plain_text_part.data(), plain_text_part.data() + plain_text_part.size());
    return std::string_view(buffer.buffer.data(), buffer.buffer.size());
}

DevicePrintParser::ArgumentValue DevicePrintParser::read_argument_from_payload(
    char type_id, std::span<const std::byte> payload_bytes, std::size_t& offset) {
    switch (type_id) {
        case 'b':  // int8_t
            return read_value_from_payload<int8_t>(payload_bytes, offset);
        case 'B':  // uint8_t
            return read_value_from_payload<uint8_t>(payload_bytes, offset);
        case 'h':  // int16_t
            return read_value_from_payload<int16_t>(payload_bytes, offset);
        case 'H':  // uint16_t
            return read_value_from_payload<uint16_t>(payload_bytes, offset);
        case 'i':  // int32_t
            return read_value_from_payload<int32_t>(payload_bytes, offset);
        case 'I':  // uint32_t
            return read_value_from_payload<uint32_t>(payload_bytes, offset);
        case 'q':  // int64_t
            return read_value_from_payload<int64_t>(payload_bytes, offset);
        case 'Q':  // uint64_t
            return read_value_from_payload<uint64_t>(payload_bytes, offset);
        case 'f':  // float
            return read_value_from_payload<float>(payload_bytes, offset);
        case 'd':  // double
            return read_value_from_payload<double>(payload_bytes, offset);
        case '?':  // bool
            return read_value_from_payload<bool>(payload_bytes, offset);
        case 'e':  // bf4_t, but stored as float
        {
            uint16_t data = read_value_from_payload<uint16_t>(payload_bytes, offset);
            uint8_t val = (data >> 8) & 0xFF;
            uint8_t exponent = data & 0xFF;
            uint32_t bit_val = convert_bfp_to_u32(tt::DataFormat::Bfp4_b, val, exponent, false);
            return *reinterpret_cast<float*>(&bit_val);
        }
        case 'E':  // bf8`_t, but stored as float
        {
            uint16_t data = read_value_from_payload<uint16_t>(payload_bytes, offset);
            uint8_t val = (data >> 8) & 0xFF;
            uint8_t exponent = data & 0xFF;
            uint32_t bit_val = convert_bfp_to_u32(tt::DataFormat::Bfp8_b, val, exponent, false);
            return *reinterpret_cast<float*>(&bit_val);
        }
        case 'w':  // bf16_t, but stored as float
        {
            uint16_t data = read_value_from_payload<uint16_t>(payload_bytes, offset);
            auto value = bfloat16_to_float(data);
            return value;
        }
        case 't':  // TileSlice, but `pad` field carried info about MAX_BYTES
        {
            TileSliceDynamic tile_slice;
            tile_slice.header = read_value_from_payload<TileSliceHostDev<0>>(payload_bytes, offset);
            tile_slice.data.resize(tile_slice.header.pad);
            for (size_t i = 0; i < tile_slice.header.pad; ++i) {
                tile_slice.data[i] = read_value_from_payload<uint8_t>(payload_bytes, offset);
            }
            return tile_slice;
        }
        default: TT_THROW("Unsupported type_id in format placeholder (read_argument_from_payload): {}", type_id);
    }
}

const DevicePrintParser::EnumInfo* DevicePrintParser::get_enum_info(std::string_view type_name) {
    if (!enum_info_loaded_) {
        load_enum_info_from_dwarf();
        enum_info_loaded_ = true;
    }
    auto it = enum_info_cache_.find(type_name);
    if (it != enum_info_cache_.end()) {
        return &it->second;
    }
    return nullptr;
}

void DevicePrintParser::load_enum_info_from_dwarf() {
    Dwarf_Debug dbg = nullptr;
    Dwarf_Error err = nullptr;

    // libdwarf can open ELF files directly by path
    int res = dwarf_init_path(
        elf_path.c_str(),
        nullptr,  // true_pathbuf
        0,        // true_pathlen
        DW_GROUPNUMBER_ANY,
        nullptr,  // errhand
        nullptr,  // errarg
        &dbg,
        &err);

    if (res != DW_DLV_OK) {
        // No DWARF info available
        return;
    }

    // Traverse all compilation units
    Dwarf_Unsigned cu_header_length = 0;
    Dwarf_Half version_stamp = 0;
    Dwarf_Off abbrev_offset = 0;
    Dwarf_Half address_size = 0;
    Dwarf_Half offset_size = 0;
    Dwarf_Half extension_size = 0;
    Dwarf_Sig8 type_signature;
    Dwarf_Unsigned typeoffset = 0;
    Dwarf_Unsigned next_cu_header = 0;
    Dwarf_Half header_cu_type = 0;
    Dwarf_Bool is_info = true;

    while (dwarf_next_cu_header_d(
               dbg,
               is_info,
               &cu_header_length,
               &version_stamp,
               &abbrev_offset,
               &address_size,
               &offset_size,
               &extension_size,
               &type_signature,
               &typeoffset,
               &next_cu_header,
               &header_cu_type,
               &err) == DW_DLV_OK) {
        Dwarf_Die cu_die = nullptr;
        if (dwarf_siblingof_b(dbg, nullptr, is_info, &cu_die, &err) != DW_DLV_OK) {
            continue;
        }

        // Recursive lambda to walk the DIE tree and collect enum types with qualified names
        std::function<void(Dwarf_Die, const std::string&)> walk_die;
        walk_die = [&](Dwarf_Die die, const std::string& parent_scope) {
            Dwarf_Half tag = 0;
            if (dwarf_tag(die, &tag, &err) != DW_DLV_OK) {
                return;
            }

            // Build the current scope name
            std::string current_scope = parent_scope;
            char* die_name = nullptr;
            bool has_name = (dwarf_diename(die, &die_name, &err) == DW_DLV_OK && die_name);

            if (tag == DW_TAG_namespace || tag == DW_TAG_class_type || tag == DW_TAG_structure_type) {
                if (has_name) {
                    if (!current_scope.empty()) {
                        current_scope += "::";
                    }
                    current_scope += die_name;
                }
            }

            if (tag == DW_TAG_enumeration_type && has_name) {
                std::string enum_qualified_name = current_scope;
                if (!enum_qualified_name.empty()) {
                    enum_qualified_name += "::";
                }
                enum_qualified_name += die_name;

                // Extract enumerator children
                EnumInfo info;
                info.type_name = enum_qualified_name;

                Dwarf_Die child = nullptr;
                if (dwarf_child(die, &child, &err) == DW_DLV_OK) {
                    while (child) {
                        Dwarf_Half child_tag = 0;
                        if (dwarf_tag(child, &child_tag, &err) == DW_DLV_OK && child_tag == DW_TAG_enumerator) {
                            char* enumerator_name = nullptr;
                            Dwarf_Signed sval = 0;
                            Dwarf_Unsigned uval = 0;
                            bool got_value = false;
                            int64_t enum_val = 0;

                            if (dwarf_diename(child, &enumerator_name, &err) == DW_DLV_OK && enumerator_name) {
                                // Try to get const_value as signed first, then unsigned
                                Dwarf_Attribute attr = nullptr;
                                if (dwarf_attr(child, DW_AT_const_value, &attr, &err) == DW_DLV_OK) {
                                    Dwarf_Half form = 0;
                                    dwarf_whatform(attr, &form, &err);
                                    if (dwarf_formsdata(attr, &sval, &err) == DW_DLV_OK) {
                                        enum_val = sval;
                                        got_value = true;
                                    } else if (dwarf_formudata(attr, &uval, &err) == DW_DLV_OK) {
                                        enum_val = static_cast<int64_t>(uval);
                                        got_value = true;
                                    }
                                    dwarf_dealloc_attribute(attr);
                                }

                                if (got_value) {
                                    info.enumerators.emplace_back(enum_val, std::string(enumerator_name));
                                }
                            }
                        }

                        Dwarf_Die sibling = nullptr;
                        if (dwarf_siblingof_b(dbg, child, is_info, &sibling, &err) != DW_DLV_OK) {
                            dwarf_dealloc_die(child);
                            break;
                        }
                        dwarf_dealloc_die(child);
                        child = sibling;
                    }
                }

                if (!info.enumerators.empty()) {
                    enum_info_cache_[enum_qualified_name] = std::move(info);
                }
            }

            // Recurse into children
            Dwarf_Die child = nullptr;
            if (dwarf_child(die, &child, &err) == DW_DLV_OK) {
                while (child) {
                    walk_die(child, (tag == DW_TAG_enumeration_type) ? parent_scope : current_scope);
                    Dwarf_Die sibling = nullptr;
                    if (dwarf_siblingof_b(dbg, child, is_info, &sibling, &err) != DW_DLV_OK) {
                        dwarf_dealloc_die(child);
                        break;
                    }
                    dwarf_dealloc_die(child);
                    child = sibling;
                }
            }
        };

        walk_die(cu_die, "");
        dwarf_dealloc_die(cu_die);
    }

    dwarf_finish(dbg);
}

}  // namespace tt::tt_metal
