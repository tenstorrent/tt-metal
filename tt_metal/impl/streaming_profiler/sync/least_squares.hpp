// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace tt::tt_metal::streaming_profiler {

// The least-squares line y = y_mean + slope * (x - x_mean).
struct LineFit {
    double x_mean = 0.0, y_mean = 0.0, slope = 0.0;
    constexpr double at(double x) const { return y_mean + slope * (x - x_mean); }
};
// Returns the least-squares line through `points`, which must hold at least two distinct x. The sums run about the
// first point, so they keep their precision when x is large next to its spread.
template <
    std::ranges::forward_range Points,
    std::invocable<std::ranges::range_reference_t<const Points>> X,
    std::invocable<std::ranges::range_reference_t<const Points>> Y>
LineFit fit_line(const Points& points, X x, Y y) {
    const auto& first = *std::ranges::begin(points);
    const double x0 = std::invoke(x, first), y0 = std::invoke(y, first);
    double sum_x = 0.0, sum_y = 0.0;
    size_t count = 0;
    for (const auto& point : points) {
        sum_x += std::invoke(x, point) - x0;
        sum_y += std::invoke(y, point) - y0;
        count++;
    }
    LineFit fit{.x_mean = x0 + sum_x / static_cast<double>(count), .y_mean = y0 + sum_y / static_cast<double>(count)};
    double sum_xx = 0.0, sum_xy = 0.0;
    for (const auto& point : points) {
        const double dx = std::invoke(x, point) - fit.x_mean;
        sum_xx += dx * dx;
        sum_xy += dx * (std::invoke(y, point) - fit.y_mean);
    }
    fit.slope = sum_xy / sum_xx;
    return fit;
}

// Solves a * x = b in place of b, where `a` is a symmetric positive-definite b.size() x b.size() matrix stored by rows.
// Its lower triangle is overwritten with its Cholesky factor.
template <std::floating_point T>
void cholesky_solve(std::span<T> a, std::span<T> b) {
    const size_t n = b.size();
    const auto at = [&](size_t row, size_t col) -> T& { return a[row * n + col]; };
    for (size_t j = 0; j < n; j++) {
        T pivot = at(j, j);
        for (size_t k = 0; k < j; k++) {
            pivot -= at(j, k) * at(j, k);
        }
        at(j, j) = std::sqrt(pivot);
        for (size_t i = j + 1; i < n; i++) {
            T below = at(i, j);
            for (size_t k = 0; k < j; k++) {
                below -= at(i, k) * at(j, k);
            }
            at(i, j) = below / at(j, j);
        }
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < i; k++) {
            b[i] -= at(i, k) * b[k];
        }
        b[i] /= at(i, i);
    }
    for (size_t i = n; i-- > 0;) {
        for (size_t k = i + 1; k < n; k++) {
            b[i] -= at(k, i) * b[k];
        }
        b[i] /= at(i, i);
    }
}

// Returns the least-squares x where each edge says x[plus(edge)] - x[minus(edge)] = value(edge). An empty end is the
// ground, which is fixed at 0, and every unknown must have a chain of edges to it. The least-squares normal equations
// are the graph's Laplacian with the ground's row and column removed.
template <
    std::ranges::forward_range Edges,
    std::invocable<std::ranges::range_reference_t<const Edges>> Plus,
    std::invocable<std::ranges::range_reference_t<const Edges>> Minus,
    std::invocable<std::ranges::range_reference_t<const Edges>> Value>
std::vector<double> solve_potential(const Edges& edges, Plus plus_of, Minus minus_of, Value value_of, size_t unknowns) {
    std::vector<double> normal(unknowns * unknowns, 0.0);
    const auto entry = [&](size_t row, size_t col) -> double& { return normal[row * unknowns + col]; };
    std::vector<double> x(unknowns, 0.0);
    for (const auto& edge : edges) {
        const std::optional<size_t> plus = std::invoke(plus_of, edge);
        const std::optional<size_t> minus = std::invoke(minus_of, edge);
        const double value = std::invoke(value_of, edge);
        if (plus) {
            entry(*plus, *plus) += 1.0;
            x[*plus] += value;
        }
        if (minus) {
            entry(*minus, *minus) += 1.0;
            x[*minus] -= value;
        }
        if (plus && minus) {
            entry(*plus, *minus) -= 1.0;
            entry(*minus, *plus) -= 1.0;
        }
    }
    cholesky_solve(std::span(normal), std::span(x));
    return x;
}

}  // namespace tt::tt_metal::streaming_profiler
