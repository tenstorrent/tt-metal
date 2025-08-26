#pragma once

// Approximation mode for SFPU operations
// Precise: high precision, slower
// Fast:    fast approximation, less accurate
enum class ApproximationMode {
	Precise = 0,
	Fast = 1
};
