// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt_stl/strong_type.hpp>
#include <tt_stl/span.hpp>
#include <optional>
#include <cstddef>
#include <cstdint>
#include <complex>

namespace tt::tt_metal::distributed::multihost {

enum class ReduceOp : std::uint8_t { SUM, MAX, MIN, PROD, LAND, LOR, BAND, BOR };

enum class DType : std::uint8_t {
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    FLOAT32,
    FLOAT64,
    BOOL,
    BYTE,
    COMPLEX_FLOAT,
    COMPLEX_DOUBLE,
};

template <typename T, typename = void>
struct dtype_of;  // intentionally incomplete – gives a clear error on unknown T

template <>
struct dtype_of<std::int8_t> {
    static constexpr DType value = DType::INT8;
};
template <>
struct dtype_of<std::uint8_t> {
    static constexpr DType value = DType::UINT8;
};
template <>
struct dtype_of<std::int16_t> {
    static constexpr DType value = DType::INT16;
};
template <>
struct dtype_of<std::uint16_t> {
    static constexpr DType value = DType::UINT16;
};
template <>
struct dtype_of<std::int32_t> {
    static constexpr DType value = DType::INT32;
};
template <>
struct dtype_of<std::uint32_t> {
    static constexpr DType value = DType::UINT32;
};
template <>
struct dtype_of<std::int64_t> {
    static constexpr DType value = DType::INT64;
};
template <>
struct dtype_of<std::uint64_t> {
    static constexpr DType value = DType::UINT64;
};
template <>
struct dtype_of<float> {
    static constexpr DType value = DType::FLOAT32;
};
template <>
struct dtype_of<double> {
    static constexpr DType value = DType::FLOAT64;
};
template <>
struct dtype_of<bool> {
    static constexpr DType value = DType::BOOL;
};
template <>
struct dtype_of<std::complex<float>> {
    static constexpr DType value = DType::COMPLEX_FLOAT;
};
template <>
struct dtype_of<std::complex<double>> {
    static constexpr DType value = DType::COMPLEX_DOUBLE;
};

template <typename T>
inline constexpr DType dtype_of_v = dtype_of<T>::value;

template <typename T, typename = void>
struct is_supported_dtype : std::false_type {};

template <typename T>
struct is_supported_dtype<T, std::void_t<decltype(dtype_of_v<T>)>> : std::true_type {};

template <typename T>
inline constexpr bool is_supported_dtype_v = is_supported_dtype<T>::value;

using Rank = tt::stl::StrongType<int, struct RankTag>;
using Tag = tt::stl::StrongType<int, struct TagTag>;
using Color = tt::stl::StrongType<int, struct ColorTag>;
using Key = tt::stl::StrongType<int, struct KeyTag>;
using Size = tt::stl::StrongType<int, struct SizeTag>;
using DistributedContextId = tt::stl::StrongType<int, struct DistributedContextIdTag>;

class DistributedException : public std::exception {
public:
    virtual Rank rank() const noexcept = 0;
    virtual int error_code() const noexcept = 0;
    virtual const std::string& message() const noexcept = 0;
    virtual const std::string& error_string() const noexcept = 0;

    const char* what() const noexcept override { return message().c_str(); }
    ~DistributedException() override = default;
};

struct Status {
    Rank source = Rank(0);
    Tag tag = Tag(0);
    int count = 0;
};

class Request {
public:
    virtual ~Request() = default;

    /// Block until the operation completes, then return its Status.
    [[nodiscard]] virtual Status wait() = 0;

    /// Poll for completion. If done, return Status; otherwise return std::nullopt.
    [[nodiscard]] virtual std::optional<Status> test() = 0;

    /// Cancel the pending operation (if supported).
    virtual void cancel() = 0;

    /// Is the request still active (i.e. not yet completed or cancelled)?
    [[nodiscard]] virtual bool active() const = 0;
};

class DistributedContext;

using RequestPtr = std::shared_ptr<Request>;
using ContextPtr = std::shared_ptr<DistributedContext>;
class DistributedContext {
public:
    static void create(int argc, char** argv);
    static const ContextPtr& get_current_world();
    static void set_current_world(const ContextPtr& ctx);

    // Returns true if the distributed context has already been initialized
    static bool is_initialized();

    // Returns a unique ID for this distributed context instance
    DistributedContextId id() const;

    //--- Topology ------------------------------------------------------------
    [[nodiscard]] virtual Rank rank() const = 0;
    [[nodiscard]] virtual Size size() const = 0;
    [[nodiscard]] virtual bool supports_fault_tolerance() const = 0;

    //--- Synchronization ----------------------------------------------------
    virtual void barrier() const = 0;

    //--- Point-to-point (blocking) -----------------------------------------
    virtual void send(tt::stl::Span<std::byte> buffer, Rank dest, Tag tag) const = 0;

    virtual void ssend(tt::stl::Span<std::byte> buffer, Rank dest, Tag tag) const = 0;

    virtual void recv(tt::stl::Span<std::byte> buffer, Rank source, Tag tag) const = 0;

    //--- Point-to-point (non-blocking) -------------------------------------
    [[nodiscard]] virtual RequestPtr isend(tt::stl::Span<std::byte> buffer, Rank dest, Tag tag) const = 0;

    [[nodiscard]] virtual RequestPtr irecv(tt::stl::Span<std::byte> buffer, Rank source, Tag tag) const = 0;

    //--- Collective operations ---------------------------------------------
    virtual void broadcast(tt::stl::Span<std::byte> buffer, Rank root) const = 0;

    virtual void gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const = 0;

    virtual void scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const = 0;

    virtual void all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const = 0;

    virtual void all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const = 0;

    /// --- Reduce functions ------------------------------------------------
    virtual void all_reduce(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const = 0;

    virtual void reduce(
        tt::stl::Span<std::byte> send_buf,
        tt::stl::Span<std::byte> recv_buf,
        ReduceOp op,
        DType dtype,
        Rank root) const = 0;

    virtual void reduce_scatter(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const = 0;

    virtual void scan(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const = 0;

    // --- Reduce functions with type deduction -------------------------------
    template <class T>
        requires is_supported_dtype_v<T>
    void all_reduce(tt::stl::Span<T> send_buf, tt::stl::Span<T> recv_buf, ReduceOp op) const {
        all_reduce(as_writable_bytes(send_buf), as_writable_bytes(recv_buf), op, dtype_of_v<T>);
    }
    template <class T>
        requires is_supported_dtype_v<T>
    void reduce(tt::stl::Span<T> send_buf, tt::stl::Span<T> recv_buf, ReduceOp op, Rank root) const {
        reduce(as_writable_bytes(send_buf), as_writable_bytes(recv_buf), op, dtype_of_v<T>, root);
    }
    template <class T>
        requires is_supported_dtype_v<T>
    void scan(tt::stl::Span<T> send_buf, tt::stl::Span<T> recv_buf, ReduceOp op) const {
        scan(as_writable_bytes(send_buf), as_writable_bytes(recv_buf), op, dtype_of_v<T>);
    }
    template <class T>
        requires is_supported_dtype_v<T>
    void reduce_scatter(tt::stl::Span<T> send_buf, tt::stl::Span<T> recv_buf, ReduceOp op) const {
        reduce_scatter(as_writable_bytes(send_buf), as_writable_bytes(recv_buf), op, dtype_of_v<T>);
    }

    //--- Communicator management -------------------------------------------
    [[nodiscard]] virtual ContextPtr duplicate() const = 0;
    [[nodiscard]] virtual ContextPtr split(Color color, Key key) const = 0;
    [[nodiscard]] virtual ContextPtr create_sub_context(tt::stl::Span<int> ranks) const = 0;
    virtual void translate_ranks_to_other_ctx(
        tt::stl::Span<int> ranks, const ContextPtr& other_ctx, tt::stl::Span<int> translated_ranks) const = 0;
    //--- Error handling -----------------------------------------------------
    virtual void abort(int error_code) const = 0;

    virtual void revoke_and_shrink() = 0;
    [[nodiscard]] virtual bool is_revoked() = 0;

    //--- Message snooping -----------------------------------------------
    // Probe for an incoming message from 'source' with 'tag'. Return the size of the message in bytes
    virtual std::size_t snoop_incoming_msg_size(Rank source, Tag tag) const = 0;

    virtual ~DistributedContext() = default;

protected:
    // This function is used to generate a unique ID for each DistributedContext instance.
    // It allows tracking which contexts are in use, and can be used for creating context specific resources.
    // This function is not thread-safe.
    static DistributedContextId generate_unique_id();
    DistributedContextId id_{0};  // Unique identifier for the context
};

}  // namespace tt::tt_metal::distributed::multihost
