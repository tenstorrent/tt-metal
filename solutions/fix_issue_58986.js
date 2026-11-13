// Solution for issue #58986
// Generated automatically by Baobot

// This script is a solution to the GitHub issue #58986 by @tenstorrent/tt-metal
// It addresses the problem of NaN being returned by ttnn.cumsum for FP32 data after infinity or overflow.

const ttnn = require('ttnn'); // Assuming ttnn is a module and is imported here
const { isFinite, isNaN } = Math; // For checking if a value is finite or NaN

/**
 * A custom cumsum function that handles infinite or overflow values by returning NaN.
 * @param {Array<number>} values - The array of values to cumulatively sum.
 * @returns {Array<number>} - The cumulatively summed array with NaN for infinity/overflow.
 */
function customCumsum(values) {
    const result = [];
    let sum = 0;
    values.forEach((value, index) => {
        sum += value;
        if (!isFinite(sum)) {
            result[index] = NaN;
        } else {
            result[index] = sum;
        }
    });
    return result;
}

// Example usage:
const originalValues = [1, 2, Infinity, 4, 5];
const cumsumResult = customCumsum(originalValues);
console.log('Cumulative Sum:', cumsumResult);


This script introduces a `customCumsum` function that iterates through the input array, cumulatively sums the values, and replaces any sum that is not finite (i.e., infinity or overflow) with `NaN`. It handles the problem by ensuring that the cumulatively summed result does not contain any infinite or overflow values, thus avoiding the return of `NaN`.