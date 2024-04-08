// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/serialization.hpp"
#include "tensor/borrowed_buffer_functions.hpp"
#include "tensor/owned_buffer_functions.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace tt {

namespace tt_metal {

namespace detail {

static constexpr std::size_t SENTINEL_VALUE = std::numeric_limits<std::size_t>::max();

void dump_owned_storage(ofstream& output_stream, const OwnedStorage& storage) {
    std::visit(
        [&output_stream]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
            const auto buffer = owned_buffer::get_as<T>(generic_buffer);
            auto size = buffer.size();
            output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
            output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
        },
        storage.buffer
    );
}

void dump_borrowed_storage(ofstream& output_stream, const BorrowedStorage& storage) {
    std::visit(
        [&output_stream]<typename T>(const borrowed_buffer::Buffer<T>& generic_buffer) {
            const auto buffer = borrowed_buffer::get_as<T>(generic_buffer);
            auto size = buffer.size();
            output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
            output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
        },
        storage.buffer
    );
}

void dump_multi_device_host_storage(ofstream& output_stream, const MultiDeviceHostStorage& storage) {
    std::size_t num_buffers = storage.buffers.size();
    output_stream.write(reinterpret_cast<const char*>(&num_buffers), sizeof(std::size_t));
    for (const auto& buffer : storage.buffers) {
        std::visit(
            [&output_stream]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
                const auto buffer = owned_buffer::get_as<T>(generic_buffer);
                auto size = buffer.size();
                output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
                output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
            }, buffer
        );
    }
    for (const auto& shape : storage.shapes) {
        output_stream.write(reinterpret_cast<const char*>(&shape), sizeof(Shape));
    }
}

template<typename T>
OwnedStorage load_owned_storage(ifstream& input_stream) {
    std::size_t size = 0;
    input_stream.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
    auto buffer = owned_buffer::create<T>(size);
    input_stream.read(reinterpret_cast<char*>(buffer.begin()), sizeof(T) * size);
    return {buffer};

}

template<typename T>
MultiDeviceHostStorage load_multi_device_host_storage(ifstream& input_stream) {
    std::size_t num_buffers = 0;
    input_stream.read(reinterpret_cast<char*>(&num_buffers), sizeof(std::size_t));

    std::vector<OwnedBuffer> buffers;

    for (std::size_t i = 0; i < num_buffers; ++i) {
        std::size_t size = 0;
        input_stream.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));

        auto buffer = owned_buffer::create<T>(size);
        input_stream.read(reinterpret_cast<char*>(buffer.begin()), sizeof(T) * size);

        buffers.push_back(std::move(buffer));
    }
    std::vector<Shape> shapes;
    for (std::size_t i = 0; i < num_buffers; ++i) {
        auto shape = Shape{};
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(Shape));
        shapes.push_back(shape);
    }

    return {buffers, shapes};
}


OwnedStorage load_owned_storage(ifstream& input_stream, DataType data_type) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B or data_type == DataType::BFLOAT4_B) {
        using T = std::uint32_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_owned_storage<T>(input_stream);
    } else {
        TT_THROW("Unsupported DataType");
    }
}


MultiDeviceHostStorage load_multi_device_host_storage(ifstream& input_stream, DataType data_type) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B) {
        using T = std::uint32_t;
        return load_multi_device_host_storage<T>(input_stream);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_multi_device_host_storage<T>(input_stream);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_multi_device_host_storage<T>(input_stream);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_multi_device_host_storage<T>(input_stream);
    } else {
        TT_THROW("Unsupported DataType");
    }
}


Storage load_storage(ifstream& input_stream, DataType data_type, StorageType storage_type) {
    if (storage_type == StorageType::MULTI_DEVICE_HOST) {
        return load_multi_device_host_storage(input_stream, data_type);
    } else {
        return load_owned_storage(input_stream, data_type);
    }
}

}

void dump_tensor(const std::string& file_name, const Tensor& tensor) {
    ofstream output_stream(file_name, ios::out | ios::binary);
    if (not output_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }

    auto shape = tensor.get_legacy_shape();
    auto data_type = tensor.get_dtype();
    auto layout = tensor.get_layout();
    auto storage_type = tensor.storage_type();

    output_stream.write(reinterpret_cast<const char*>(&detail::SENTINEL_VALUE), sizeof(std::size_t));
    output_stream.write(reinterpret_cast<const char*>(&VERSION_ID), sizeof(std::uint8_t));
    output_stream.write(reinterpret_cast<const char*>(&shape), sizeof(Shape));
    output_stream.write(reinterpret_cast<const char*>(&data_type), sizeof(DataType));
    output_stream.write(reinterpret_cast<const char*>(&layout), sizeof(Layout));
    output_stream.write(reinterpret_cast<const char*>(&storage_type), sizeof(StorageType));

    std::visit(
        [&output_stream](const auto& storage) {

            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                detail::dump_owned_storage(output_stream, storage);
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                detail::dump_borrowed_storage(output_stream, storage);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            }
            else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            }
            else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                detail::dump_multi_device_host_storage(output_stream, storage);
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
}

Tensor load_tensor(const std::string& file_name) {
    ifstream input_stream(file_name, ios::in | ios::binary);
    if (not input_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }

    std::size_t read_sentinel;
    input_stream.read(reinterpret_cast<char*>(&read_sentinel), sizeof(read_sentinel));
    if (read_sentinel == detail::SENTINEL_VALUE) {
        std::uint8_t version_id;
        input_stream.read(reinterpret_cast<char*>(&version_id), sizeof(version_id));
        if (version_id != VERSION_ID) {
            throw std::runtime_error(fmt::format("Unsupported version_id: {}", version_id));
        }
        auto shape = Shape{};
        DataType data_type;
        Layout layout;
        StorageType storage_type;
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(Shape));
        input_stream.read(reinterpret_cast<char*>(&data_type), sizeof(DataType));
        input_stream.read(reinterpret_cast<char*>(&layout), sizeof(Layout));
        input_stream.read(reinterpret_cast<char*>(&storage_type), sizeof(StorageType));

        auto storage = detail::load_storage(input_stream, data_type, storage_type);
        return Tensor(std::move(storage), shape, data_type, layout);

    } else {
        input_stream.seekg(0, ios::beg); // No sentinel found, assume it's an older format and rewind

        auto shape = Shape{};
        DataType data_type;
        Layout layout;
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(Shape));
        input_stream.read(reinterpret_cast<char*>(&data_type), sizeof(DataType));
        input_stream.read(reinterpret_cast<char*>(&layout), sizeof(Layout));

        auto storage = detail::load_owned_storage(input_stream, data_type);
        return Tensor(std::move(storage), shape, data_type, layout);
    }
}


}  // namespace tt_metal

}  // namespace tt
