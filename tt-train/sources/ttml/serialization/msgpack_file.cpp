// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "msgpack_file.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <fstream>
#define MSGPACK_NO_BOOST
#include <fstream>
#include <msgpack.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
    namespace adaptor {

    // Custom adaptor for std::variant
    template <typename... Types>
    struct pack<std::variant<Types...>> {
        template <typename Stream>
        packer<Stream>& operator()(msgpack::packer<Stream>& o, const std::variant<Types...>& v) const {
            // Pack the index of the active type and the value
            o.pack_array(2);
            o.pack(v.index());
            std::visit([&o](const auto& val) { o.pack(val); }, v);
            return o;
        }
    };

    template <typename... Types>
    struct convert<std::variant<Types...>> {
        msgpack::object const& operator()(msgpack::object const& o, std::variant<Types...>& v) const {
            if (o.type != msgpack::type::ARRAY || o.via.array.size != 2) {
                throw std::runtime_error(
                    "Invalid object type. Expected array of size 2. Where first value is the  type index and second is "
                    "our object.");
            }

            std::size_t index = o.via.array.ptr[0].as<std::size_t>();

            auto& obj = o.via.array.ptr[1];

            // Helper lambda to set the variant based on index
            bool success = set_variant_by_index(index, obj, v);
            if (!success) {
                throw std::runtime_error(fmt::format(
                    "Cannot convert object to variant. Possible reason: type mismatch. Object index: {}", index));
            }

            return o;
        }

    private:
        template <std::size_t N = 0>
        bool set_variant_by_index(std::size_t index, msgpack::object const& obj, std::variant<Types...>& v) const {
            if constexpr (N < sizeof...(Types)) {
                if (index == N) {
                    using T = std::variant_alternative_t<N, std::variant<Types...>>;
                    T val;
                    obj.convert(val);
                    v = std::move(val);
                    return true;
                } else {
                    return set_variant_by_index<N + 1>(index, obj, v);
                }
            } else {
                throw std::runtime_error(fmt::format("Invalid index for variant type. Index: {}", index));
            }
        }
    };

    }  // namespace adaptor
}  // namespace MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
}  // namespace msgpack

namespace ttml::serialization {
class MsgPackFile::Impl {
public:
    // Methods to store different types
    void put(std::string_view key, bool value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, char value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, int value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, float value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, double value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, uint32_t value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, size_t value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, const std::string& value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, std::string_view value) {
        m_data[std::string(key)] = std::string(value);
    }

    // Overloads for std::span
    void put(std::string_view key, std::span<const uint8_t> value) {
        m_data[std::string(key)] = std::vector<uint8_t>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const int> value) {
        m_data[std::string(key)] = std::vector<int>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const float> value) {
        m_data[std::string(key)] = std::vector<float>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const double> value) {
        m_data[std::string(key)] = std::vector<double>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const uint32_t> value) {
        m_data[std::string(key)] = std::vector<uint32_t>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const std::string> value) {
        m_data[std::string(key)] = std::vector<std::string>(value.begin(), value.end());
    }

    void put(std::string_view key, const ValueType& value) {
        m_data[std::string(key)] = value;
    }

    // Serialization method
    void serialize(const std::string& filename) {
        // Create a buffer for packing
        msgpack::sbuffer sbuf;

        // Pack the data into the buffer
        msgpack::pack(sbuf, m_data);

        // Write the buffer to a file
        std::ofstream ofs(filename, std::ios::binary);
        if (ofs.is_open()) {
            ofs.write(sbuf.data(), static_cast<std::streamsize>(sbuf.size()));
            ofs.close();
        } else {
            throw std::runtime_error("Unable to open file for writing: " + filename);
        }
    }

    // Deserialization method
    void deserialize(const std::string& filename) {
        // Read the file content into a string buffer
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Unable to open file for reading: " + filename);
        }
        std::string buffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        ifs.close();

        // Unpack the buffer into msgpack object
        msgpack::object_handle handle = msgpack::unpack(buffer.data(), buffer.size());

        // Convert the msgpack object to the desired type
        msgpack::object obj = handle.get();

        // Clear existing data
        m_data.clear();

        // Convert object to m_data
        obj.convert(m_data);
    }

    // Methods to get values
    bool get(std::string_view key, bool& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, char& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, int& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, float& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, double& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, uint32_t& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, size_t& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, std::string& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, std::vector<uint8_t>& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, std::vector<int>& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, std::vector<float>& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, std::vector<double>& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, std::vector<uint32_t>& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, std::vector<std::string>& value) const {
        return get_value(key, value);
    }

    bool get(std::string_view key, ValueType& value) const {
        return get_value(key, value);
    }

private:
    std::unordered_map<std::string, ValueType> m_data;

    // Helper function to get value from m_data
    template <typename T>
    bool get_value(std::string_view key, T& value) const {
        auto it = m_data.find(std::string(key));
        if (it != m_data.end()) {
            if (const auto* pval = std::get_if<T>(&(it->second))) {
                value = *pval;
                return true;
            } else {
                throw std::runtime_error(fmt::format("Type mismatch for key: {}", key));
            }
        } else {
            // Key not found
            throw std::runtime_error(fmt::format("Key not found: {}", key));
        }
    }
    template <>
    bool get_value<ValueType>(std::string_view key, ValueType& value) const {
        auto it = m_data.find(std::string(key));
        if (it != m_data.end()) {
            value = it->second;
            return true;
        } else {
            // Key not found
            throw std::runtime_error(fmt::format("Key not found: {}", key));
        }
    }
};

MsgPackFile::MsgPackFile() : m_impl(std::make_unique<Impl>()) {
}

MsgPackFile::~MsgPackFile() = default;

MsgPackFile::MsgPackFile(MsgPackFile&&) noexcept = default;

void MsgPackFile::put(std::string_view key, bool value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, char value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, int value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, float value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, double value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, uint32_t value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, size_t value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, std::string_view value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, std::span<const uint8_t> value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, std::span<const int> value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, std::span<const float> value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, std::span<const double> value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, std::span<const uint32_t> value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, std::span<const std::string> value) {
    m_impl->put(key, value);
}

void MsgPackFile::put(std::string_view key, const char* value) {
    put(key, std::string_view(value));
}

void MsgPackFile::put(std::string_view key, const ValueType& value) {
    m_impl->put(key, value);
}

void MsgPackFile::serialize(const std::string& filename) {
    m_impl->serialize(filename);
}

void MsgPackFile::deserialize(const std::string& filename) {
    m_impl->deserialize(filename);
}

void MsgPackFile::get(std::string_view key, bool& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, char& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, int& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, float& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, double& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, uint32_t& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, size_t& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, std::string& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, std::vector<uint8_t>& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, std::vector<int>& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, std::vector<float>& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, std::vector<double>& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, std::vector<uint32_t>& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, std::vector<std::string>& value) const {
    m_impl->get(key, value);
}

void MsgPackFile::get(std::string_view key, ValueType& value) const {
    m_impl->get(key, value);
}

}  // namespace ttml::serialization
