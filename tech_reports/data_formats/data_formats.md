# Data Formats

## Conversion from bfloat16 to block-floats

<img width="958" alt="image" src="https://github.com/user-attachments/assets/03a2e044-f14e-4d59-86b5-a04c193ad3fa">

## Block-float-8, block-float-4, block-float-2

<img width="961" alt="image" src="https://github.com/user-attachments/assets/e1f54311-e6a6-48f3-9030-192de985b2ce">

## Mantissa Rounding
When converting from a higher precision to lower precision data format, the mantissa is rounded to the nearest. If the value to round is tied, then it rounds to the nearest even value for the mantissa. For example, when converting from float32 to bfloat8, we want to round 23 bits of mantissa for float32 to 7 bits of mantissa for bfloat8. However, we also explicitly store the hidden bit of 1 for bfloat8, so we are really rounding to 6 bits total. Consider the following 23 bits of mantissa:

<img width="803" alt="image" src="https://github.com/user-attachments/assets/d8d17ad0-8679-406c-9587-1661f2319965" />

To get the 7 bits of mantissa for bfloat8, we want to keep 6 bits of the original 23-bit mantissa and store the additional hidden bit at the most significant bit (MSB). The least significant bit (LSB) of the 6-bit mantissa to keep is known as the guard bit, which we use to round to the nearest even (if there is a tie). In other implementations or literature, the MSB of the round value is also known as the round bit with the remaining bits denoted as the sticky bit(s), but the result is the same. In host code, the rounding is done with the following process:

<img width="1041" alt="image" src="https://github.com/user-attachments/assets/9adaf40a-750c-4c5c-8ec6-c7ff2fbb2bf9" />

To handle exponent sharing, the mantissa is first normalized prior to rounding if the exponent is different from the shared exponent. If there is an overflow in the mantissa when we round up, we do not recompute the max shared exponent and re-normalize across the 16 numbers. Instead, the mantissa is set to the max value (ie. all 1's). For the other block float formats, the same process applies but with the corresponding number of bits for the mantissa and round value.
