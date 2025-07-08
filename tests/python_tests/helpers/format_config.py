# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class DataFormatInfo:
    """
    A helper class that encapsulates metadata for a data format.

    Attributes:
        name (str): A human-readable name for the data format.
        byte_size (int): The size in bytes of one unit of the data format.
    """

    def __init__(self, name: str, byte_size: int):
        self.name = name
        self.byte_size = byte_size


class DataFormat(Enum):
    """
    An enumeration of data formats supported by the LLKs.
    Holds format name and byte size, and is extendable.
    """

    Float16 = DataFormatInfo("Float16", 2)
    Float16_b = DataFormatInfo("Float16_b", 2)
    Bfp8_b = DataFormatInfo("Bfp8_b", 1)
    Float32 = DataFormatInfo("Float32", 4)
    Int32 = DataFormatInfo("Int32", 4)
    Tf32 = DataFormatInfo("Tf32", 3)
    UInt32 = DataFormatInfo("UInt32", 4)
    UInt16 = DataFormatInfo("UInt16", 2)
    Int8 = DataFormatInfo("Int8", 1)
    UInt8 = DataFormatInfo("UInt8", 1)

    @property
    def size(self) -> int:
        """Returns the byte size of the data format."""
        return self.value.byte_size

    def __str__(self) -> str:
        """Returns the string representation of the data format."""
        return self.value.name

    def is_integer(self) -> bool:
        """Checks if the data format is an integer type."""
        return self in {
            DataFormat.Int32,
            DataFormat.UInt32,
            DataFormat.UInt16,
            DataFormat.Int8,
            DataFormat.UInt8,
        }

    def is_32_bit(self) -> bool:
        """Checks if the data format is a 32-bit type."""
        return self in {DataFormat.Float32, DataFormat.Int32, DataFormat.UInt32}

    def is_exponent_B(self) -> bool:
        """Checks if the data format is an exponent B format."""
        return self in {
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
            DataFormat.Tf32,
            DataFormat.Float32,
        }


@dataclass
class FormatConfig:
    """
    A data class that holds configuration details for formats passed to LLKs.tions).

    Attributes:
    unpack_A_src (DataFormat): The source format for source register A in the Unpacker, which is the format of our data in L1.
    unpack_A_dst (DataFormat): The destination format for source register A in the Unpacker, which is the format of our data in the source register.
    unpack_B_src (Optional[DataFormat]): The source format for source register B in the Unpacker, which is the format of our data in L1. Optional; defaults to `unpack_A_src` if `same_src_format=True`.
    unpack_B_dst (Optional[DataFormat]): The destination format for source register B in the Unpacker, which is the format of our data in the source register. Optional; defaults to `unpack_A_dst` if `same_src_format=True`.
    pack_src (DataFormat): The source format for the Packer.
    pack_dst (DataFormat): The destination format for the Packer.
    math (DataFormat): The format used for _llk_math_ functions.

    Optional Parameters:
    same_src_format (bool): If `True`, the formats for source registers A and B will be the same for unpack operations.
    If `False`, source registers A and B have different formats formats must be specified. Defaults to `True`.

    unpack_B_src (Optional[DataFormat]): The source format for source register B in the Unpacker which is the format of our data in L1, used only if `same_src_format=False` i.e when source regosters don't share the same formats we distinguish source register A and B formats.
    unpack_B_dst (Optional[DataFormat]): The destination format for source register B in the Unpacker, which is the format of our data in src register used only if `same_src_format=False` i.e when source registers don't share the same formats we distinguish source register A and B formats.

    Example:
    >>> formats = FormatConfig(
    >>>     unpack_A_src=DataFormat.Float32,
    >>>     unpack_A_dst=DataFormat.Float16,
    >>>     pack_src=DataFormat.Float16,
    >>>     pack_dst=DataFormat.Float32,
    >>>     math=DataFormat.Float32    # same_src_format defaults to True, thus our source registers have same formats and we don't need to define formats for source register B
    >>> )
    >>> print(formats.unpack_A_src)
    DataFormat.Float32
    >>> print(formats.unpack_B_src)
    DataFormat.Float32                 # B formats match A if same_src_format=True
    """

    unpack_A_src: DataFormat
    unpack_A_dst: DataFormat
    unpack_B_src: Optional[DataFormat]
    unpack_B_dst: Optional[DataFormat]
    pack_src: DataFormat
    pack_dst: DataFormat
    math: DataFormat

    def __init__(
        self,
        unpack_A_src: DataFormat,
        unpack_A_dst: DataFormat,
        pack_src: DataFormat,
        pack_dst: DataFormat,
        math: DataFormat,
        same_src_format: bool = True,  # if True, source registers A and B have the same formats, don't need to pass next 2 parameters
        # if our src registers have the same formats, then we only pass 5 formats into the FormatConfig object
        # and we set unpack_B_src and unpack_B_dst the same format as input formats for source register A
        unpack_B_src: Optional[DataFormat] = None,
        unpack_B_dst: Optional[DataFormat] = None,
    ):

        self.unpack_A_src = unpack_A_src
        self.unpack_A_dst = unpack_A_dst
        self.pack_src = pack_src
        self.pack_dst = pack_dst
        self.math = math
        if same_src_format:
            self.unpack_B_src = unpack_A_src
            self.unpack_B_dst = unpack_A_dst
        else:
            if unpack_B_src is None or unpack_B_dst is None:
                raise ValueError(
                    "When same_src_format is False, both unpack_B_src and unpack_B_dst must be provided."
                )
            self.unpack_B_src = unpack_B_src
            self.unpack_B_dst = unpack_B_dst

    @property
    def output_format(self) -> DataFormat:
        return self.pack_dst

    @property
    def input_format(self) -> DataFormat:
        return self.unpack_A_src


@dataclass
class InputOutputFormat:
    """
    A data class that holds configuration details for formats passed to LLKs.
    This class is used to hold input and output DataFormat that the client wants to test.
    They are used for format inference model to infer the rest of the formats for the LLk pipeline, instead of the user.
    """

    input: DataFormat
    output: DataFormat

    def __init__(self, input_format: DataFormat, output_format: DataFormat):
        self.input = input_format
        self.output = output_format

    @property
    def output_format(self) -> DataFormat:
        return self.output

    @property
    def input_format(self) -> DataFormat:
        return self.input


def create_formats_for_testing(formats: List[Tuple[DataFormat]]) -> List[FormatConfig]:
    """
    A function that creates a list of FormatConfig objects from a list of DataFormat objects that client wants to test.
    This function is useful for creating a list of FormatConfig objects for testing multiple formats combinations
    and cases which the user has specifically defined and wants to particularly test instead of a full format flush.

    Args:
    formats (List[Tuple[DataFormat]]): A list of tuples of DataFormat objects for which FormatConfig objects need to be created.

    Returns:
    List[FormatConfig]: A list of FormatConfig objects created from the list of DataFormat objects passed as input.

    Example:
    >>> formats = [(DataFormat.Float16, DataFormat.Float32, DataFormat.Float16, DataFormat.Float32, DataFormat.Float32)]
    >>> format_configs = create_formats_for_testing(formats)
    >>> print(format_configs[0].unpack_A_src)
    DataFormat.Float16
    >>> print(format_configs[0].unpack_B_src)
    DataFormat.Float16
    """
    format_configs = []
    for format_tuple in formats:
        if len(format_tuple) == 5:
            format_configs.append(
                FormatConfig(
                    unpack_A_src=format_tuple[0],
                    unpack_A_dst=format_tuple[1],
                    pack_src=format_tuple[2],
                    pack_dst=format_tuple[3],
                    math=format_tuple[4],
                )
            )
        else:
            format_configs.append(
                FormatConfig(
                    unpack_A_src=format_tuple[0],
                    unpack_A_dst=format_tuple[1],
                    unpack_B_src=format_tuple[2],
                    unpack_B_dst=format_tuple[3],
                    pack_src=format_tuple[4],
                    pack_dst=format_tuple[5],
                    math=format_tuple[6],
                    same_src_format=False,
                )
            )
    return format_configs


def is_dest_acc_needed(format: InputOutputFormat) -> bool:
    """
    This function is called when a format configuration for input and output is called without dest accumulation.
    If the input-output combination is an outlier that is not supported when dest accumulation is on
    then the data format inference model will turn dest accumulation off for this combination to work.

    We must notify the user that this has happened and cheange the test output to reflect this.
    """
    return (
        format.input_format in [DataFormat.Bfp8_b, DataFormat.Float16_b]
        and format.output_format == DataFormat.Float16
    )
