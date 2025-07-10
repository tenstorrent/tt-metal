from boltz.data.types import Record
from boltz.data.filter.dynamic.filter import DynamicFilter


class SizeFilter(DynamicFilter):
    """A filter that filters structures based on their size."""

    def __init__(self, min_chains: int = 1, max_chains: int = 300) -> None:
        """Initialize the filter.

        Parameters
        ----------
        min_chains : int
            The minimum number of chains allowed.
        max_chains : int
            The maximum number of chains allowed.

        """
        self.min_chains = min_chains
        self.max_chains = max_chains

    def filter(self, record: Record) -> bool:
        """Filter structures based on their resolution.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        num_chains = record.structure.num_chains
        num_valid = sum(1 for chain in record.chains if chain.valid)
        return num_chains <= self.max_chains and num_valid >= self.min_chains
