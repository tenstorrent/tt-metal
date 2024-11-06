#  replace <[]> to <[none]>
# look for | Unknown  | Unknown    | N/A   | and isolate those s0 shapes
md_data = """
|   2 | Tensor<[1, 1, 1, 42]> self = , Tensor other = -6.0                        | Done     | Done       | True  |
|   3 | Tensor<[1, 1, 1, 42]> self = , Tensor other = 0.5                         | Done     | Done       | True  |
|   4 | Tensor<[1, 1, 1, 42]> self = , Tensor other = 1                           | Done     | Done       | True  |
|   5 | Tensor<[1, 1, 1, 42]> self = , Tensor other = 1.0                         | Done     | Done       | True  |
|   6 | Tensor<[1, 1, 1, 42]> self = , Tensor other = 2                           | Done     | Done       | True  |
|   7 | Tensor<[1, 1, 1024]> self = , Tensor other = 1.0                          | Unknown  | Done       | True  |
|  10 | Tensor<[1, 1, 1]> self = , Tensor other = 0.8999999985098839              | Done     | Done       | True  |
|  11 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9043478220701218              | Done     | Done       | True  |
|  12 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9086956530809402              | Done     | Done       | True  |
|  13 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9090909063816071              | Done     | Done       | True  |
|  14 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9130434766411781              | Done     | Done       | True  |
|  15 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9142857119441032              | Done     | Done       | True  |
|  16 | Tensor<[1, 1, 1]> self = , Tensor other = 0.917391300201416               | Done     | Done       | True  |
|  17 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9181818142533302              | Done     | Done       | True  |
|  18 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9217391312122345              | Done     | Done       | True  |
|  19 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9260869547724724              | Done     | Done       | True  |
|  20 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9272727221250534              | Done     | Done       | True  |
|  21 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9285714253783226              | Done     | Done       | True  |
|  22 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9304347857832909              | Done     | Done       | True  |
|  23 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9347826093435287              | Done     | Done       | True  |
|  24 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9363636374473572              | Done     | Done       | True  |
|  25 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9391304366290569              | Done     | Done       | True  |
|  26 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9428571425378323              | Done     | Done       | True  |
|  27 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9434782639145851              | Done     | Done       | True  |
|  28 | Tensor<[1, 1, 1]> self = , Tensor other = 0.94545454159379                | Done     | Done       | True  |
|  29 | Tensor<[1, 1, 1]> self = , Tensor other = 0.947826087474823               | Done     | Done       | True  |
|  30 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9521739110350609              | Done     | Done       | True  |
|  31 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9545454569160938              | Done     | Done       | True  |
|  32 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9565217345952988              | Done     | Done       | True  |
|  33 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9571428559720516              | Done     | Done       | True  |
|  34 | Tensor<[1, 1, 1]> self = , Tensor other = 0.960869561880827               | Done     | Done       | True  |
|  35 | Tensor<[1, 1, 1]> self = , Tensor other = 0.963636364787817               | Done     | Done       | True  |
|  36 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9652173891663551              | Done     | Done       | True  |
|  37 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9695652164518833              | Done     | Done       | True  |
|  38 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9714285712689161              | Done     | Done       | True  |
|  39 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9727272726595402              | Done     | Done       | True  |
|  40 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9739130418747663              | Done     | Done       | True  |
|  41 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9782608672976494              | Done     | Done       | True  |
|  42 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9818181823939085              | Done     | Done       | True  |
|  43 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9826086945831776              | Done     | Done       | True  |
|  44 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9857142856344581              | Done     | Done       | True  |
|  45 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9869565209373832              | Done     | Done       | True  |
|  46 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9909090911969543              | Done     | Done       | True  |
|  47 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9913043472915888              | Done     | Done       | True  |
|  48 | Tensor<[1, 1, 1]> self = , Tensor other = 0.9956521736457944              | Done     | Done       | True  |
|  49 | Tensor<[1, 1, 1]> self = , Tensor other = 1e-06                           | Unknown  | Done       | True  |
|  50 | Tensor<[1, 1, 224, 224]> self = , Tensor other = -0.030000000000000027    | Done     | Done       | True  |
|  51 | Tensor<[1, 1, 224, 224]> self = , Tensor other = -0.08799999999999997     | Done     | Done       | True  |
|  52 | Tensor<[1, 1, 224, 224]> self = , Tensor other = -0.18799999999999994     | Done     | Done       | True  |
|  53 | Tensor<[1, 1, 3072]> self = , Tensor other = 1.0                          | Unknown  | Done       | True  |
|  55 | Tensor<[1, 1, 32, 1]> self = , Tensor other = -6.0                        | Done     | Done       | True  |
|  56 | Tensor<[1, 1, 32, 1]> self = , Tensor other = 0.5                         | Done     | Done       | True  |
|  57 | Tensor<[1, 1, 32, 1]> self = , Tensor other = 1                           | Done     | Done       | True  |
|  58 | Tensor<[1, 1, 32, 1]> self = , Tensor other = 1.0                         | Done     | Done       | True  |
|  59 | Tensor<[1, 1, 32, 1]> self = , Tensor other = 2                           | Done     | Done       | True  |
|  60 | Tensor<[1, 1, 4096]> self = , Tensor other = 1.0                          | Unknown  | Done       | True  |
|  62 | Tensor<[1, 1, 40]> self = , Tensor other = 1e-06                          | Done     | Done       | True  |
|  68 | Tensor<[1, 10, 1]> self = , Tensor other = 1e-06                          | Done     | Done       | True  |
|  73 | Tensor<[1, 1024, 1, 1]> self = , Tensor other = 0.0                       | Unknown  | Fallback   | True  |
|  74 | Tensor<[1, 1024, 1, 1]> self = , Tensor other = 1e-05                     | None     | Fallback   | True  |
|  89 | Tensor<[1, 10]> self = , Tensor other = 0                                 | Done     | Done       | True  |
|  90 | Tensor<[1, 10]> self = , Tensor other = 1                                 | Done     | Done       | True  |
| 114 | Tensor<[1, 12, 3072]> self = , Tensor other = 1.0                         | Done     | Done       | True  |
| 125 | Tensor<[1, 128, 1, 1]> self = , Tensor other = 0.0                        | Unknown  | Fallback   | True  |
| 126 | Tensor<[1, 128, 1, 1]> self = , Tensor other = 1e-05                      | None     | Fallback   | True  |
| 150 | Tensor<[1, 14, 3072]> self = , Tensor other = 1.0                         | Done     | Done       | True  |
| 157 | Tensor<[1, 15, 1024]> self = , Tensor other = 1.0                         | Done     | Done       | True  |
| 159 | Tensor<[1, 15, 1]> self = , Tensor other = 1e-06                          | Done     | Done       | True  |
| 218 | Tensor<[1, 19]> self = , Tensor other = 2                                 | None     | Fallback   | True  |
| 219 | Tensor<[1, 1]> self = , Tensor other = 0                                  | Unknown  | Done       | True  |
| 220 | Tensor<[1, 1]> self = , Tensor other = 16                                 | Unknown  | Done       | True  |
| 221 | Tensor<[1, 1]> self = , Tensor other = 2                                  | Unknown  | Done       | True  |
| 224 | Tensor<[1, 2048, 1, 1]> self = , Tensor other = 0.0                       | Unknown  | Fallback   | True  |
| 225 | Tensor<[1, 2048, 1, 1]> self = , Tensor other = 1e-05                     | None     | Fallback   | True  |
| 238 | Tensor<[1, 23, 1]> self = , Tensor other = 1e-06                          | Done     | Done       | True  |
| 254 | Tensor<[1, 256, 1, 1]> self = , Tensor other = 0.0                        | Unknown  | Fallback   | True  |
| 255 | Tensor<[1, 256, 1, 1]> self = , Tensor other = 1e-05                      | None     | Fallback   | True  |
| 305 | Tensor<[1, 32, 6144]> self = , Tensor other = 1                           | Done     | Done       | True  |
| 306 | Tensor<[1, 32, 6144]> self = , Tensor other = 1.0                         | Done     | Done       | True  |
| 342 | Tensor<[1, 45, 3072]> self = , Tensor other = 1.0                         | Unknown  | Done       | True  |
| 355 | Tensor<[1, 5, 4096]> self = , Tensor other = 1.0                          | Unknown  | Done       | True  |
| 360 | Tensor<[1, 512, 1, 1]> self = , Tensor other = 0.0                        | Unknown  | Fallback   | True  |
| 361 | Tensor<[1, 512, 1, 1]> self = , Tensor other = 1e-05                      | None     | Fallback   | True  |
| 384 | Tensor<[1, 59]> self = , Tensor other = 2                                 | Done     | Done       | True  |
| 398 | Tensor<[1, 64, 1, 1]> self = , Tensor other = 0.0                         | Unknown  | Fallback   | True  |
| 399 | Tensor<[1, 64, 1, 1]> self = , Tensor other = 1e-05                       | None     | Fallback   | True  |
| 435 | Tensor<[1, 7, 3072]> self = , Tensor other = 1.0                          | Done     | Done       | True  |
| 481 | Tensor<[1, 9, 128]> self = , Tensor other = 1.0                           | Done     | Done       | True  |
| 483 | Tensor<[1, 9, 16384]> self = , Tensor other = 1.0                         | Done     | Done       | True  |
| 486 | Tensor<[1, 9, 3072]> self = , Tensor other = 1.0                          | Done     | Done       | True  |
| 488 | Tensor<[1, 9, 4096]> self = , Tensor other = 1.0                          | Done     | Done       | True  |
| 491 | Tensor<[1, 9, 8192]> self = , Tensor other = 1.0                          | Done     | Done       | True  |
| 506 | Tensor<[10, 10]> self = , Tensor other = 0                                | Done     | Done       | True  |
| 507 | Tensor<[10, 10]> self = , Tensor other = 8                                | Done     | Done       | True  |
| 510 | Tensor<[100]> self = , Tensor other = 0.0                                 | Unknown  | Done       | True  |
| 511 | Tensor<[1066]> self = , Tensor other = 0.5                                | Unknown  | Done       | True  |
| 512 | Tensor<[10]> self = , Tensor other = 0.5                                  | Done     | Done       | True  |
| 514 | Tensor<[120]> self = , Tensor other = 0.5                                 | Done     | Done       | True  |
| 515 | Tensor<[128]> self = , Tensor other = 0.5                                 | Done     | Done       | True  |
| 516 | Tensor<[12]> self = , Tensor other = 0.0                                  | None     | Fallback   | True  |
| 518 | Tensor<[136]> self = , Tensor other = 0.0                                 | Unknown  | Done       | True  |
| 519 | Tensor<[14]> self = , Tensor other = 0.0                                  | None     | Fallback   | True  |
| 520 | Tensor<[15, 15]> self = , Tensor other = 0                                | Done     | Done       | True  |
| 521 | Tensor<[15, 15]> self = , Tensor other = 8                                | Done     | Done       | True  |
| 527 | Tensor<[160]> self = , Tensor other = 0.5                                 | Done     | Done       | True  |
| 528 | Tensor<[16]> self = , Tensor other = 0.0                                  | None     | Fallback   | True  |
| 529 | Tensor<[17, 17]> self = , Tensor other = 0                                | Unknown  | Done       | True  |
| 530 | Tensor<[17, 17]> self = , Tensor other = 16                               | Unknown  | Done       | True  |
| 531 | Tensor<[19]> self = , Tensor other = 0.5                                  | Done     | Done       | True  |
| 532 | Tensor<[1]> self = , Tensor other = 0.5                                   | Done     | Done       | True  |
| 536 | Tensor<[2, 2]> self = , Tensor other = 0                                  | Unknown  | Done       | True  |
| 537 | Tensor<[2, 2]> self = , Tensor other = 16                                 | Unknown  | Done       | True  |
| 544 | Tensor<[20]> self = , Tensor other = 0.5                                  | Done     | Done       | True  |
| 546 | Tensor<[23]> self = , Tensor other = 0.0                                  | Done     | Done       | True  |
| 547 | Tensor<[24, 24]> self = , Tensor other = 160                              | None     | Fallback   | True  |
| 548 | Tensor<[240]> self = , Tensor other = 0.5                                 | Done     | Done       | True  |
| 550 | Tensor<[28]> self = , Tensor other = 0.0                                  | None     | Fallback   | True  |
| 551 | Tensor<[2]> self = , Tensor other = 0.5                                   | Done     | Done       | True  |
| 552 | Tensor<[300]> self = , Tensor other = 0.5                                 | Done     | Done       | True  |
| 553 | Tensor<[30]> self = , Tensor other = 0.5                                  | Done     | Done       | True  |
| 554 | Tensor<[320]> self = , Tensor other = 0.5                                 | Done     | Done       | True  |
| 558 | Tensor<[32]> self = , Tensor other = 0.0                                  | Done     | Done       | True  |
| 560 | Tensor<[38]> self = , Tensor other = 0.5                                  | Done     | Done       | True  |
| 561 | Tensor<[3]> self = , Tensor other = 0.5                                   | Done     | Done       | True  |
| 566 | Tensor<[40]> self = , Tensor other = 0.0                                  | Done     | Done       | True  |
| 567 | Tensor<[40]> self = , Tensor other = 0.5                                  | Done     | Done       | True  |
| 568 | Tensor<[480]> self = , Tensor other = 0.5                                 | Done     | Done       | True  |
| 569 | Tensor<[50]> self = , Tensor other = 0.0                                  | Unknown  | Done       | True  |
| 570 | Tensor<[56]> self = , Tensor other = 0.0                                  | None     | Fallback   | True  |
| 572 | Tensor<[5]> self = , Tensor other = 0.5                                   | Done     | Done       | True  |
| 573 | Tensor<[60]> self = , Tensor other = 0.5                                  | Done     | Done       | True  |
| 579 | Tensor<[640]> self = , Tensor other = 0.5                                 | Done     | Done       | True  |
| 580 | Tensor<[64]> self = , Tensor other = 0.0                                  | Done     | Done       | True  |
| 581 | Tensor<[68]> self = , Tensor other = 0.0                                  | Unknown  | Done       | True  |
| 582 | Tensor<[7]> self = , Tensor other = 0.0                                   | None     | Fallback   | True  |
| 583 | Tensor<[800]> self = , Tensor other = 0.5                                 | Unknown  | Done       | True  |
| 584 | Tensor<[80]> self = , Tensor other = 0.5                                  | Done     | Done       | True  |
"""

left = """
| 591 | Tensor<[]> self = , Tensor other = 1                                      | None     | Fallback   | True  |

| 533 | Tensor<[2*s0]> self = , Tensor other = 0.0                                | Unknown  | Unknown    | N/A   |
| 534 | Tensor<[2*s1]> self = , Tensor other = 0.0                                | Unknown  | Unknown    | N/A   |
| 535 | Tensor<[2*s2]> self = , Tensor other = 0.0                                | Unknown  | Unknown    | N/A   |
| 593 | Tensor<[s0 + 1, s0 + 1]> self = , Tensor other = 0                        | Unknown  | Unknown    | N/A   |
| 594 | Tensor<[s0 + 1, s0 + 1]> self = , Tensor other = 16                       | Unknown  | Unknown    | N/A   |

"""
import re


def extract_tensor_shapes(md_string):
    """
    Extracts tensor shapes for "self" and values for "other" from the input string.
    Returns a list of dictionaries with the shape of "self" and value of "other".
    """
    shapes = []

    # Regular expressions to match "self" tensor shape and "other" value
    self_pattern = re.compile(r"Tensor<\[(.*?)\]> self")
    other_pattern = re.compile(r"Tensor other = ([\d\.\-e]+)")

    # Split by lines and process each line
    for line in md_string.splitlines():
        self_match = self_pattern.search(line)
        other_match = other_pattern.search(line)

        # Extract "self" shape
        if self_match:
            # Check if there's a single dimension or multiple
            dimensions = self_match.group(1)
            if "," in dimensions:
                self_shape = list(map(int, dimensions.split(",")))
            else:
                self_shape = [int(dimensions)]  # Wrap in a list for single dimension
        else:
            self_shape = None

        # Extract "other" value
        if other_match:
            other_value = float(other_match.group(1))  # Convert scalar to float
        else:
            other_value = None

        # If both "self" and "other" are found, add to the list
        if self_shape is not None and other_value is not None:
            shapes.append({"self": self_shape, "other": other_value})

    return shapes


# Extract shapes from the markdown string
shapes = extract_tensor_shapes(md_data)
print(shapes)

# # Print the extracted shapes - self or other
# for shape in shapes:
#     vec = shape["other"]
#     print(f"{vec},")
