from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional

from numpy.random import RandomState

from boltz.data.types import Record


@dataclass
class Sample:
    """A sample with optional chain and interface IDs.

    Attributes
    ----------
    record : Record
        The record.
    chain_id : Optional[int]
        The chain ID.
    interface_id : Optional[int]
        The interface ID.
    """

    record: Record
    chain_id: Optional[int] = None
    interface_id: Optional[int] = None


class Sampler(ABC):
    """Abstract base class for samplers."""

    @abstractmethod
    def sample(self, records: List[Record], random: RandomState) -> Iterator[Sample]:
        """Sample a structure from the dataset infinitely.

        Parameters
        ----------
        records : List[Record]
            The records to sample from.
        random : RandomState
            The random state for reproducibility.

        Yields
        ------
        Sample
            A data sample.

        """
        raise NotImplementedError
