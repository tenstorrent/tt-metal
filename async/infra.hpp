#pragma once

#include <cassert>
#include <format>
#include <functional>
#include <future>
#include <iostream>
#include <sstream>
#include <thread>
#include <tuple>
#include <variant>

#include "reflect.hpp"
#include "thread_safe_queue.hpp"

// Infra code

struct Program {
    char* data;

    ~Program() { free(data); }
};

struct Device {
    static std::size_t COUNTER;

    // In real code, there is an abstraction called work executor
    thread_safe_queue_t<std::function<void()>> worker_queue;
    std::thread worker_thread;
    bool run = true;
    std::size_t id = COUNTER++;

    Device() {
        worker_thread = std::thread([this] {
            while (this->run) {
                auto computation = this->worker_queue.pop_front();
                computation();
            }
        });
    }

    void enqueue(Program& program) {
        constexpr std::size_t size = 1'000'000'000;
        program.data = static_cast<char*>(malloc(size));
        memset(program.data, 5, size);
    }

    void push_computation(std::function<void()>&& computation) {
        this->worker_queue.emplace_back(std::move(computation));
    }

    ~Device() {
        while (not this->worker_queue.empty()) {
        }
        this->run = false;
        this->worker_queue.emplace_back([] { std::cout << "Shutting down device worker thread" << std::endl; });
        this->worker_thread.join();
    }
};

std::size_t Device::COUNTER = 0;

enum class Dtype {
    bfloat16,
};

struct Shape {
    std::vector<std::size_t> dims;

    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "{";
        for (auto index = 0; index < shape.dims.size(); index++) {
            os << shape.dims.at(index);
            if (index < shape.dims.size() - 1) {
                os << ",";
            }
        }
        os << "}";
        return os;
    }
};

struct DeviceStorage {
    std::shared_ptr<Device> device = nullptr;
};

struct MultiDeviceStorage {
    std::vector<std::shared_ptr<Device>> devices;
};

using Storage = std::variant<DeviceStorage, MultiDeviceStorage>;

struct Tensor {
    Shape shape;
    Dtype dtype;
    Storage storage;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "Tensor{";
        os << "shape=" << tensor.shape;
        os << "}";
        return os;
    }

    std::shared_future<Tensor> to_async() const {
        return std::async(std::launch::async, [tensor = *this] { return tensor; });
    }
};

struct input_operation_t {
    Tensor arg;
    Tensor operator()() const { return arg; }
    Tensor invoke_without_allocation() const { return arg; }
};

template <typename concrete_operation_t>
concept DeviceOperation = requires { typename concrete_operation_t::device_operation_t; };

template <typename, template <typename...> typename>
constexpr bool is_specialization_v = false;

template <template <typename...> typename T, typename... Ts>
constexpr bool is_specialization_v<T<Ts...>, T> = true;

template <typename T>
using async_container_t = std::shared_future<T>;

template <typename arg_t>
constexpr bool is_async_class() {
    // TODO: should we allow std::future?
    return is_specialization_v<std::decay_t<arg_t>, std::shared_future>;
}

template <typename... args_t>
constexpr bool is_async_call() {
    return (false or ... or is_async_class<args_t>());
}

template <typename... args_t>
concept AsyncCall = is_async_call<args_t...>();

auto map_async_to_sync(auto&&... args) {
    return std::tuple{[](auto&& arg) {
        if constexpr (is_async_class<decltype(arg)>()) {
            arg.wait();
            return arg.get();
        } else {
            return arg;
        }
    }(args)...};
}

std::shared_ptr<Device> get_device(auto&& arg, auto&&... args) {
    // Handle different types
    return std::get<DeviceStorage>(arg.storage).device;
}

struct dispatcher_t {
    // In real code, there is an abstraction called work executor
    thread_safe_queue_t<std::function<void()>> worker_queue;
    std::thread worker_thread;
    bool run = true;

    dispatcher_t() {
        worker_thread = std::thread([this] {
            while (this->run) {
                auto computation = this->worker_queue.pop_front();
                computation();
            }
        });
    }

    void push_computation(std::function<void()>&& computation) {
        this->worker_queue.emplace_back(std::move(computation));
    }

    ~dispatcher_t() {
        while (not this->worker_queue.empty()) {
        }
        this->run = false;
        this->worker_queue.emplace_back([] { std::cout << "Shutting down dispatcher worker thread" << std::endl; });
        this->worker_thread.join();
    }
};

static dispatcher_t DISPATCHER;

template <reflect::fixed_string name>
struct scoped_timer_t {
    decltype(std::chrono::system_clock::now()) start;

    scoped_timer_t() : start{std::chrono::system_clock::now()} {};

    ~scoped_timer_t() {
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - this->start);
        std::cout << std::format("{} finished in {} us\n", std::string{name}, elapsed.count());
    }
};

const Tensor& get_first_tensor(auto&& arg, auto&&... args) {
    if constexpr (std::same_as<Tensor, std::decay_t<decltype(arg)>>) {
        return arg;
    } else {
        return get_first_tensor(std::forward<decltype(args)>(args)...);
    }
}

auto get_shard_args(std::size_t shard_index, auto&&... args) {
    return std::tuple{[shard_index](auto&& arg) {
        if constexpr (std::same_as<Tensor, std::decay_t<decltype(arg)>>) {
            auto& multi_device_storage = std::get<MultiDeviceStorage>(arg.storage);
            return Tensor{arg.shape, arg.dtype, DeviceStorage{multi_device_storage.devices.at(shard_index)}};
        } else {
            return arg;
        }
    }(args)...};
}

static auto split_multi_device_inputs_into_shards(auto&&... args) {
    /*
    // Algorithm for split_multi_device_inputs_into_shards
        1. Find multi-device tensor from arguments
        2. Get number of shards
        3. Make sure all tensors have the same number of shards
        4. Loop over all shards of every tensor and populate shard_args
    */

    const auto& first_tensor = get_first_tensor(std::forward<decltype(args)>(args)...);
    auto& multi_device_storage = std::get<MultiDeviceStorage>(first_tensor.storage);
    auto num_shards = multi_device_storage.devices.size();

    std::vector<std::tuple<std::decay_t<decltype(args)>...>> output;
    for (auto index = 0; index < num_shards; index++) {
        auto shard_args = get_shard_args(index, std::forward<decltype(args)>(args)...);
        output.push_back(shard_args);
    }
    return output;
}

static Tensor create_multi_device_tensor_from_shards(std::vector<Tensor>& output_shards) {
    std::vector<std::shared_ptr<Device>> devices;
    for (const auto& shard : output_shards) {
        devices.push_back(std::get<DeviceStorage>(shard.storage).device);
    }
    return Tensor{output_shards.at(0).shape, output_shards.at(0).dtype, MultiDeviceStorage{devices}};
}

static std::vector<Tensor> create_multi_device_tensor_from_shards(std::vector<std::vector<Tensor>>& output) {
    std::vector<Tensor> combined_output;
    for (auto& output_shards : output) {
        combined_output.emplace_back(create_multi_device_tensor_from_shards(output_shards));
    }
    return combined_output;
}

template <
    std::size_t index,
    typename... Ts,
    typename tensor_return_value_t = std::decay_t<decltype(std::get<index>(std::declval<std::tuple<Ts...>>()))>>
std::shared_future<tensor_return_value_t> get_from_async(std::shared_future<std::tuple<Ts...>> future_tuple) {
    std::shared_ptr<std::promise<tensor_return_value_t>> output_promise =
        std::make_shared<std::promise<tensor_return_value_t>>();
    auto output_future = std::shared_future{output_promise->get_future()};
    DISPATCHER.push_computation([future_tuple, output_promise = std::move(output_promise)] mutable {
        output_promise->set_value(std::get<index>(future_tuple.get()));
    });
    return output_future;
}

template <typename T>
std::shared_future<T> get_from_async(std::shared_future<std::vector<T>> future_vector, std::size_t index) {
    std::shared_ptr<std::promise<T>> output_promise = std::make_shared<std::promise<T>>();
    auto output_future = std::shared_future{output_promise->get_future()};
    DISPATCHER.push_computation([future_vector, index, output_promise = std::move(output_promise)] mutable {
        output_promise->set_value(future_vector.get()[index]);
    });
    return output_future;
}

template <reflect::fixed_string name, typename concrete_operation_t>
struct operation_t {
    auto invoke(auto&&... args) const
        requires DeviceOperation<concrete_operation_t> and (not AsyncCall<decltype(args)...>)
    {
        using device_operation_t = concrete_operation_t::device_operation_t;
        using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;

        const auto& first_tensor = get_first_tensor(std::forward<decltype(args)>(args)...);
        if (std::holds_alternative<DeviceStorage>(first_tensor.storage)) {
            auto device = get_device(std::forward<decltype(args)>(args)...);

            std::shared_ptr<std::promise<tensor_return_value_t>> promise =
                std::make_shared<std::promise<tensor_return_value_t>>();
            auto future = promise->get_future();

            device->push_computation([device = device, args..., promise = std::move(promise)]() mutable {
                std::cout << std::format(
                    "started  running {} device operation on device {}\n", std::string{name}, device->id);

                auto&& [operation_attributes, tensor_args] =
                    concrete_operation_t::map_args_to_device_operation(std::forward<decltype(args)>(args)...);
                auto output = device_operation_t::create_output_tensors(operation_attributes, tensor_args);
                promise->set_value(output);
                std::cout << std::format(
                    "unblock dispatcher thread running device operation {} from device {}\n",
                    std::string{name},
                    device->id);

                auto program = device_operation_t::create_program(operation_attributes, tensor_args, output);
                device->enqueue(program);
                std::cout << std::format(
                    "finished running {} device operation on device {}\n", std::string{name}, device->id);
            });
            return future.get();
        } else {
            // Split multi-device storage into shards and launch each shard on the corresponding device

            auto& multi_device_storage = std::get<MultiDeviceStorage>(first_tensor.storage);
            auto num_shards = multi_device_storage.devices.size();

            // Preallocate shard_futures
            std::vector<std::shared_future<tensor_return_value_t>> shard_futures;
            shard_futures.reserve(num_shards);

            // Launch each shard
            for (auto shard_index = 0; shard_index < num_shards; shard_index++) {
                auto shard_args = get_shard_args(shard_index, std::forward<decltype(args)>(args)...);
                auto device = std::apply(
                    [](auto&&... args) { return get_device(std::forward<decltype(args)>(args)...); }, shard_args);
                assert(device != nullptr && "Device can't be null");

                std::shared_ptr<std::promise<tensor_return_value_t>> shard_promise =
                    std::make_shared<std::promise<tensor_return_value_t>>();
                shard_futures.push_back(shard_promise->get_future());

                device->push_computation(
                    [device = device, shard_args, shard_promise = std::move(shard_promise)]() mutable {
                        std::cout << std::format(
                            "started  running {} device operation shard on device {}\n", std::string{name}, device->id);

                        auto&& [operation_attributes, tensor_args] = std::apply(
                            [](auto&&... args) {
                                return concrete_operation_t::map_args_to_device_operation(
                                    std::forward<decltype(args)>(args)...);
                            },
                            shard_args);
                        auto output = device_operation_t::create_output_tensors(operation_attributes, tensor_args);
                        shard_promise->set_value(output);
                        std::cout << std::format(
                            "unblock dispatcher thread running device operation {} from device {}\n",
                            std::string{name},
                            device->id);

                        auto program = device_operation_t::create_program(operation_attributes, tensor_args, output);
                        device->enqueue(program);
                        std::cout << std::format(
                            "finished running {} device operation shard on device {}\n", std::string{name}, device->id);
                    });
            }

            // Combine shards into a multi-device storage
            std::vector<tensor_return_value_t> outputs;
            for (auto& shard_future : shard_futures) {
                outputs.push_back(shard_future.get());
            }
            return create_multi_device_tensor_from_shards(outputs);
        }
    }

    auto invoke(auto&&... args) const
        requires(not DeviceOperation<concrete_operation_t>) and (not AsyncCall<decltype(args)...>)
    {
        return concrete_operation_t::operator()(std::forward<decltype(args)>(args)...);
    }

    auto invoke(auto&&... args) const
        requires DeviceOperation<concrete_operation_t> and AsyncCall<decltype(args)...>
    {
        using device_operation_t = concrete_operation_t::device_operation_t;
        using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;

        std::shared_ptr<std::promise<tensor_return_value_t>> output_promise =
            std::make_shared<std::promise<tensor_return_value_t>>();
        auto output_future = std::shared_future{output_promise->get_future()};
        DISPATCHER.push_computation([this, args..., output_promise = std::move(output_promise)]() mutable {
            auto sync_args_tuple = map_async_to_sync(std::forward<decltype(args)>(args)...);

            std::cout << std::format("launching {} shards on DISPATCHER thread\n", std::string{name});

            auto output = std::apply(
                [this](auto&&... args) { return this->invoke(std::forward<decltype(args)>(args)...); },
                sync_args_tuple);
            output_promise->set_value(output);
        });
        return output_future;
    }

    auto invoke(auto&&... args) const
        requires(not DeviceOperation<concrete_operation_t>) and AsyncCall<decltype(args)...>
    {
        using tensor_return_value_t = decltype(std::apply(
            [](auto&&... args) { return concrete_operation_t::operator()(std::forward<decltype(args)>(args)...); },
            map_async_to_sync(std::forward<decltype(args)>(args)...)));

        std::shared_ptr<std::promise<tensor_return_value_t>> output_promise =
            std::make_shared<std::promise<tensor_return_value_t>>();
        auto output_future = std::shared_future{output_promise->get_future()};
        DISPATCHER.push_computation([args..., output_promise = std::move(output_promise)] mutable {
            std::cout << std::format("launching {} shards on DISPATCHER thread\n", std::string{name});

            auto sync_args_tuple = map_async_to_sync(std::forward<decltype(args)>(args)...);
            auto output = std::apply(
                [](auto&&... args) { return concrete_operation_t::operator()(std::forward<decltype(args)>(args)...); },
                map_async_to_sync(std::forward<decltype(args)>(args)...));
            output_promise->set_value(std::move(output));
        });
        return output_future;
    }

    auto operator()(auto&&... args) const {
        constexpr auto is_device_operation = static_cast<bool>(DeviceOperation<concrete_operation_t>);
        std::cout << std::format(
            "Started  operator():  {} {}\n", std::string{name}, (is_device_operation ? " (device)" : " (composite)"));
        //           << std::endl;
        auto output = invoke(std::forward<decltype(args)>(args)...);
        std::cout << std::format(
            "Finished  operator():  {} {}\n", std::string{name}, (is_device_operation ? " (device)" : " (composite)"));
        return output;
    }
};

template <reflect::fixed_string name, typename concrete_operation_t>
constexpr auto register_operation() {
    return operation_t<name, concrete_operation_t>{};
}
