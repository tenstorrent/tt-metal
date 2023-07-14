#pragma once

#include "third_party/magic_enum/magic_enum.hpp"

#include <boost/core/demangle.hpp>
#include <fmt/core.h>

#include <experimental/type_traits>
#include <ostream>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace tt {
namespace stl {
namespace reflection {

using Attributes = std::vector<std::tuple<std::string, std::string>>;

static std::ostream& operator<<(std::ostream& os, const Attributes& attributes) {
    os << "(";
    for (auto index = 0; index < attributes.size(); index++) {
        auto&& [key, value] = attributes[index];
        os << key << "=" << value;
        if (index != attributes.size() - 1) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

namespace detail {
template<typename T>
using has_attributes_t = decltype(std::declval<T>().attributes());
}

template<typename T>
typename std::enable_if_t<std::experimental::is_detected<detail::has_attributes_t, T>::value, std::ostream>&
operator<<(std::ostream& os, const T& object) {
    static_assert(std::is_same_v<decltype(object.attributes()), Attributes>);
    os << boost::core::demangle(typeid(T).name());
    os << object.attributes();
    return os;
}

template<typename T>
typename std::enable_if_t<std::is_enum<T>::value, std::ostream>&
operator<<(std::ostream& os, const T& value) {
    os << magic_enum::enum_type_name<T>() << "::" << magic_enum::enum_name(value);
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& optional) {
    if (optional.has_value()) {
        os << optional.value();
    } else {
        os << "std::nullopt";
    }
    return os;
}

template<typename ... Ts>
std::ostream& operator<<(std::ostream& os, const std::variant<Ts...>& variant) {
    std::visit(
        [&os](const auto& value) {
            os << value;
        },
        variant
    );
    return os;
}

template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& array) {
    os << "{";
    for (auto index = 0; index < array.size(); index++) {
        const auto& element = array[index];
        os << element;
        if (index != array.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
    os << "{";
    for (auto index = 0; index < vector.size(); index++) {
        const auto& element = vector[index];
        os << element;
        if (index != vector.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

}  // namespace reflection
}  // namespace stl
}  // namespace tt


template <>
struct fmt::formatter<tt::stl::reflection::Attributes> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const tt::stl::reflection::Attributes& attributes, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << attributes;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<std::experimental::is_detected<tt::stl::reflection::detail::has_attributes_t, T>::value>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const T& object, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << object;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<std::is_enum<T>::value>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const T& value, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << value;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename T>
struct fmt::formatter<std::optional<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const std::optional<T>& optional, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << optional;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename ... Ts>
struct fmt::formatter<std::variant<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const std::variant<Ts...>& variant, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << variant;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename T, std::size_t N>
struct fmt::formatter<std::array<T, N>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const std::array<T, N>& array, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << array;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename T>
struct fmt::formatter<std::vector<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const std::vector<T>& vector, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << vector;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};
